# https://stackoverflow.com/a/46327978

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import starsmashertools.tk_backend.widgets
import tkinter
from tkinter import ttk

import starsmashertools
import os
import pathlib

IMAGE_DIR = os.path.join(
    starsmashertools.SOURCE_DIRECTORY,
    'icons',
    'matplotlib',
)
PLAY_IMAGE = os.path.join(IMAGE_DIR, 'play.png')
PAUSE_IMAGE = os.path.join(IMAGE_DIR, 'pause.png')
FASTFORWARD_IMAGE = os.path.join(IMAGE_DIR, 'fastforward.png')
REWIND_IMAGE = os.path.join(IMAGE_DIR, 'rewind.png')
SKIPFORWARD_IMAGE = os.path.join(IMAGE_DIR, 'skipforward.png')
SKIPBACK_IMAGE = os.path.join(IMAGE_DIR, 'skipback.png')





        


class Player(FuncAnimation, object):
    """""" # Prevent bad docstring inheritance
    def __init__(
            self,
            fig,
            func,
            frames=None,
            init_func=None,
            fargs=None,
            save_count=None,
            mini=0,
            maxi=100,
            pos=(0., 1.),
            **kwargs
    ):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.fig = fig
        self.func = func
        self._runs = None
        self._forwards = None
        self.setup(pos)
        self.runs = True
        super(Player, self).__init__(
            self.fig,
            self.update,
            frames = self.play(), 
            init_func = init_func,
            fargs = fargs,
            save_count = save_count,
            **kwargs
        )
        self._slider_drag = False
        self._in_onestep = False

    @property
    def forwards(self): return self._forwards

    @forwards.setter
    def forwards(self, value):
        if self._forwards == value: return
        if value:
            self.set_sunken(self.buttons['fast forward'])
            self.set_flat(self.buttons['rewind'])
        else:
            self.set_sunken(self.buttons['rewind'])
            self.set_flat(self.buttons['fast forward'])
        self._forwards = value

    @property
    def runs(self): return self._runs

    @runs.setter
    def runs(self, value):
        if self._runs == value: return
        if value:
            self.set_sunken(self.buttons['play'])
            self.buttons['play']._image_file = PAUSE_IMAGE
            if self.fig.canvas.manager is not None:
                self.fig.canvas.manager.toolbar._set_image_for_button(self.buttons['play'])
            self.slider.state(['disabled'])
            self.slider.config(takefocus=False)
        else:
            self.set_flat(self.buttons['play'])
            self.buttons['play']._image_file = PLAY_IMAGE
            if self.fig.canvas.manager is not None:
                self.fig.canvas.manager.toolbar._set_image_for_button(self.buttons['play'])
            self.slider.state(['!disabled'])
            self.slider.config(takefocus=True)
        self._runs = value

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def toggle_button(self, button, from_ = None):
        if from_ is None: current = button['relief']
        else: current = from_['relief']
        if current == 'flat': self.set_sunken(button)
        else: self.set_flat(button)
    
    def set_sunken(self, button):
        button.config(relief = 'sunken', overrelief = 'sunken')
    def set_flat(self, button):
        button.config(relief = 'flat', overrelief = 'flat')
    
    # Unpauses
    def start(self):
        self.runs = True
        self.event_source.start()

    # Pauses
    def stop(self, event = None):
        self.runs = False
        self.event_source.stop()
    
    def toggle(self, event = None):
        if self.runs: self.stop(event = event)
        else: self.start()

    def forward(self):
        self.forwards = True
    
    def backward(self):
        self.forwards = False
    
    def oneforward(self, event=None):
        self.forwards = True
        self.stop()
        self.onestep()
        
    def onebackward(self, event=None):
        self.forwards = False
        self.stop()
        self.onestep()

    def onestep(self):
        self._in_onestep = True
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider_variable.set(self.i)
        self.fig.canvas.draw_idle()
        self._in_onestep = False

    def show_save_figure_dialog(self, *args, **kwargs):
        self.pause()
        def save_animation(*args, **kwargs):
            widget = self.fig.canvas.get_tk_widget()
            title = widget.winfo_toplevel().title().replace(' ','_')
            saveas = tkinter.filedialog.asksaveasfilename(
                master = widget.master,
                title = 'Save the figure',
                defaultextension = '',
                initialdir = os.path.expanduser(
                    matplotlib.rcParams['savefig.directory'],
                ),
                initialfile = pathlib.Path(self.fig.canvas.get_default_filename()).stem + '.gif',
            )
            if saveas:
                self.show_progressbar()
                self.message('Saving...', time = None)
                def progress(current, *args, **kwargs):
                    self.progress.set(current)
                    self.fig.canvas.get_tk_widget().winfo_toplevel().update_idletasks()
                try:
                    self.save(
                        saveas,
                        progress_callback = lambda val,*a,**k: self.progress.set(val),# progress,
                    )
                except Exception as e:
                    tkinter.messagebox.showerror("Error saving file", str(e))
                self.hide_progressbar()
                self.message('Saved')

        def save_image(*args, **kwargs):
            self.fig.canvas.manager.toolbar.save_figure()
            self.message('Saved')
        
        starsmashertools.tk_backend.widgets.SaveFigureDialog(
            self.fig.canvas.get_tk_widget(),
            image_callback = save_image,
            animation_callback = save_animation,
        )

        
    def setup(self, pos):
        import matplotlib
        import starsmashertools
        import os
        import matplotlib.backends._backend_tk
        
        # Now we need to force the use of Tk
        matplotlib.use('tkagg', force = True)

        toolbar = self.fig.canvas.manager.toolbar

        toolbar._buttons['Save'].config(command = self.show_save_figure_dialog)

        spacer = toolbar._Spacer()
        
        frame = tkinter.Frame(toolbar)
        
        self.buttons = {}
        self.buttons['rewind'] = matplotlib.backends._backend_tk.NavigationToolbar2Tk._Button(frame,
            'REWIND', REWIND_IMAGE, False, self.backward,
        )
        self.buttons['skip back'] = matplotlib.backends._backend_tk.NavigationToolbar2Tk._Button(frame,
            "SKIPBACK", SKIPBACK_IMAGE, False, self.onebackward,
        )
        self.buttons['play'] = matplotlib.backends._backend_tk.NavigationToolbar2Tk._Button(frame,
            "PLAY", PLAY_IMAGE, False, self.toggle,
        )
        self.buttons['skip forward'] = matplotlib.backends._backend_tk.NavigationToolbar2Tk._Button(frame,
            "SKIPFORWARD", SKIPFORWARD_IMAGE, False, self.oneforward,
        )
        self.buttons['fast forward'] = matplotlib.backends._backend_tk.NavigationToolbar2Tk._Button(frame,
            "FASTFORWARD", FASTFORWARD_IMAGE, False, self.forward,
        )

        slider_frame = tkinter.Frame(frame)

        def on_slider_changed(new_value):
            # This is called when the user clicks and drags the slider
            # We snap the value to the nearest integer
            new_value = round(float(new_value))
            self.slider_variable.set(new_value)

        self.slider_variable = tkinter.IntVar()
        self.slider_label_variable = tkinter.StringVar()
        self.slider = ttk.Scale(
            slider_frame,
            from_=self.min,
            to=self.max,
            orient='horizontal',
            length = None,
            variable = self.slider_variable,
            command = on_slider_changed,
        )
        def is_valid(action, value):
            action = int(action)
            if action == 0: # Delete
                if not value.strip(): return True

            if value.strip() == '-':
                return True
                
            try:
                int(value)
            except: return False
            
            value = int(value)
            value = min(max(value, self.min), self.max)
            self.slider_variable.set(value)
            return True
            
        
        vcmd = (slider_frame.register(is_valid), '%d', '%P')
        label = ttk.Entry(
            slider_frame,
            width = 5,
            justify = 'center',
            validate = 'key',
            validatecommand=vcmd,
        )

        def setentry(*args, **kwargs):
            try:
                label.delete(0,'end')
                label.insert('end', self.slider_variable.get())
            except: pass
        self.slider_variable.trace_add(
            'write',
            setentry,
        )

        self.message_variable = tkinter.StringVar()
        self.message_label = ttk.Label(
            frame,
            textvariable = self.message_variable,
        )

        self.progress = tkinter.IntVar()
        self.progressbar = starsmashertools.tk_backend.widgets.Progressbar(
            frame,
            maximum = self.max,
            variable = self.progress,
            textvariable = self.message_variable,
            width = 100,
        )
        

        self.slider.pack(side = 'left')
        label.pack(side = 'left',padx=(10,0))
        slider_frame.pack(side = 'left', expand = False, anchor = 'w', padx = 5)

        self.message_label.pack(side = 'left', padx=5)
        frame.pack(side='left', anchor='w')

        
        self.slider_variable.set(self.i)
        
        self.slider.state(['disabled'])
        self.slider.config(takefocus=False)

        self.slider_variable.trace_add('write', self.on_slider_variable_changed)
        self.slider.bind("<ButtonPress-1>", self.on_button_press, '+')
        self.slider.bind("<B1-Motion>", self.on_drag, '+')
        self.slider.bind("<ButtonRelease-1>", self.on_slider_click, '+')
        
        self.forwards = True

    def show_progressbar(self, *args, **kwargs):
        self.message_label.pack_forget()
        self.progressbar.pack(side='left',fill = 'y')
        self.fig.canvas.get_tk_widget().winfo_toplevel().update_idletasks()

    def hide_progressbar(self, *args, **kwargs):
        self.progressbar.pack_forget()
        self.message_label.pack(side='left')
        self.fig.canvas.get_tk_widget().winfo_toplevel().update_idletasks()
        
    def message(self, text : str, time : int | type(None) = 2000):
        widget = self.fig.canvas.get_tk_widget()
        
        if hasattr(self, '_message_after'):
            widget.after_cancel(self._message_after)
            del self._message_after
        self.message_variable.set(text)
        
        
        widget.update_idletasks()
        if time is not None:
            self._message_after = widget.after(
                time,
                lambda *a,**k: self.message_variable.set(''),
            )

    def on_button_press(self, *args, **kwargs):
        if 'disabled' in self.slider.state(): return
        self._slider_drag = False
    def on_drag(self,*args,**kwargs):
        if 'disabled' in self.slider.state(): return
        self._slider_drag = True

    def on_slider_variable_changed(self, *args, **kwargs):
        if self._in_onestep: return
        self.set_pos()
        
    def on_slider_click(self, event):
        if 'disabled' in self.slider.state(): return

        if not self._slider_drag:
            # wasn't dragging, so we must want to jump to a value
            self.stop()
            width = self.slider.winfo_width()
            value = int(round((self.max - self.min) / float(width) * event.x))
            self.slider.set(value)
            return
        
        self._slider_drag = False
        self.set_pos()
        
    def set_pos(self, *args, **kwargs):
        if self._slider_drag: return
        self.i = int(self.slider_variable.get())
        self.func(self.i)

    def update(self,i):
        self.slider_variable.set(i)
