# https://stackoverflow.com/a/46327978

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

import starsmashertools
import os

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
            self.fig.canvas.manager.toolbar._set_image_for_button(self.buttons['play'])
            self.slider.state(['disabled'])
            self.slider.config(takefocus=False)
        else:
            self.set_flat(self.buttons['play'])
            self.buttons['play']._image_file = PLAY_IMAGE
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

    def setup(self, pos):
        import matplotlib
        import starsmashertools
        import os
        import tkinter
        from tkinter import ttk
        import matplotlib.backends._backend_tk
        
        # Now we need to force the use of Tk
        matplotlib.use('tkagg', force = True)

        toolbar = self.fig.canvas.manager.toolbar

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

        frame.pack(side='left', anchor='w')

        self.slider_variable = tkinter.IntVar()
        self.slider = ttk.Scale(
            frame,
            from_=self.min,
            to=self.max,
            orient='horizontal',
            length = None,
            variable = self.slider_variable,
        )
        self.slider_variable.set(self.i)
        self.slider.pack(side='left',expand=False, anchor='w', padx=5)

        self.slider.state(['disabled'])
        self.slider.config(takefocus=False)

        self.slider_variable.trace_add('write', self.on_slider_variable_changed)
        self.slider.bind("<ButtonPress-1>", self.on_button_press, '+')
        self.slider.bind("<B1-Motion>", self.on_drag, '+')
        self.slider.bind("<ButtonRelease-1>", self.on_slider_click, '+')
        
        self.forwards = True

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