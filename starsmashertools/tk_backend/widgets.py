r"""
Custom tkinter widgets are defined here.
"""
import tkinter
from tkinter import ttk
from tkinter import simpledialog


class SaveFigureDialog(simpledialog.Dialog, object):
    def __init__(
            self,
            *args,
            image_callback = None,
            animation_callback = None,
            **kwargs
    ):
        self.image_callback = image_callback
        self.animation_callback = animation_callback
        super(SaveFigureDialog, self).__init__(*args, **kwargs)
        try:
            self.winfo_toplevel().title('Save as')
        except: pass

    def body(self, *args, **kwargs):
        label = ttk.Label(
            self,
            text = 'Save as an animation, or only the current frame?',
        )
        label.pack(expand = True, anchor='c', fill = 'both', padx = 5)
        
    def buttonbox(self):
        frame = tkinter.Frame(self)
        animation_button = ttk.Button(
            frame,
            text = 'Animation',
            command = self.on_animation_press,
        )

        image_button = ttk.Button(
            frame,
            text = 'Current Frame',
            command = self.on_image_press,
        )
        animation_button.pack(side='left',padx=5)
        image_button.pack(side = 'right',padx=5)
        frame.pack(side = 'bottom', anchor='c', expand = False, fill = 'y', pady=5)

    def on_animation_press(self, *args, **kwargs):
        self.destroy()
        self.update()
        if self.animation_callback is not None:
            self.animation_callback(*args, **kwargs)

    def on_image_press(self, *args, **kwargs):
        self.destroy()
        self.update()
        if self.image_callback is not None:
            self.image_callback(*args, **kwargs)


class Progressbar(ttk.Frame, object):
    style_initialized = False
    
    def __init__(
            self,
            master,
            *args,
            variable = None,
            textvariable = None,
            maximum = 100,
            style = "Progressbar.TFrame",
            borderwidth = 1,
            relief = 'sunken',
            **kwargs
    ):
        ttkstyle = ttk.Style()
        if not Progressbar.style_initialized:
            # https://stackoverflow.com/a/56678946/4954083
            ttkstyle.map(
                "Progressbar.TFrame",
                background = [
                    ('!disabled', ttkstyle.lookup("TProgressbar",'troughcolor',state=['!disabled'])),
                    ('disabled', ttkstyle.lookup("TProgressbar",'troughcolor',state=['disabled'])),
                ],
                darkcolor = [
                    ('!disabled', ttkstyle.lookup("TProgressbar",'darkcolor',state=['!disabled'])),
                    ('disabled', ttkstyle.lookup("TProgressbar",'darkcolor',state=['disabled'])),
                ],
                lightcolor = [
                    ('!disabled', ttkstyle.lookup("TProgressbar",'lightcolor',state=['!disabled'])),
                    ('disabled', ttkstyle.lookup("TProgressbar",'lightcolor',state=['disabled'])),
                ],
                bordercolor = [
                    ('!disabled', ttkstyle.lookup("TProgressbar",'bordercolor',state=['!disabled'])),
                    ('disabled', ttkstyle.lookup("TProgressbar",'bordercolor',state=['disabled'])),
                ],
            )
            Progressbar.style_initialized = True
        
        self.font = kwargs.pop('font', None)

        if variable is None: variable = tkinter.DoubleVar(value = 0)
        variable.trace_add('write', self._update_progress)

        self.value = variable
        
        if textvariable is not None:
            textvariable.trace_add(
                'write',
                lambda *a, **k: self.set_text(textvariable.get()),
            )

        self.maximum = maximum
        super(Progressbar, self).__init__(
            master,
            *args,
            style = style,
            borderwidth = borderwidth,
            relief = relief,
            **kwargs
        )
        self._state = ["!disabled"]
        self.state(self._state)

        self._canvas = tkinter.Canvas(self,bd=0,highlightthickness=0)
        self._canvas.place(relx=0,rely=0,relwidth=1,relheight=1)
        self._progress_rectangle = self._canvas.create_rectangle(0, 0, 0, 0)
        self._text = self._canvas.create_text(0, 0, text="", font=self.font)

        self.bind("<Configure>", self._resize_canvas, add="+")
        self._canvas.addtag_all("all")

    def _get_state(self):
        result = self._state
        if "normal" in self._state:
            result[result.index("normal")] = "!disabled"
        return result

    def configure(self, *args, **kwargs):
        try:
            self.state([kwargs.pop('state',None)])
            if 'value' in kwargs.keys(): self.value.set(kwargs.pop('value'))
            self.font = kwargs.pop('font', self.font)
            super(Progressbar,self).configure(*args,**kwargs)
            self.event_generate("<Configure>")
        except tkinter.TclError as e:
            if "invalid command name" in str(e): return
            raise(e)

    def get_progress_color(self, *args, **kwargs):
        return ttk.Style().lookup(
            "TProgressbar", 'background', state=self._get_state(),
        )
    def get_background_color(self, *args, **kwargs):
        return ttk.Style().lookup(
            "TProgressbar", "troughcolor", state=self._get_state(),
        )
        
    def set_text(self,new_text):
        try:
            self._canvas.itemconfig(self._text, text=new_text)
        except tkinter.TclError as e:
            if "invalid command name" in str(e): return
            raise(e)

    def _update_progress(self, *args, **kwargs):
        try:
            width = self.winfo_width()
            height = self.winfo_height()
            x1 = int(self.value.get()/float(self.maximum)  * width)
        
            # Update the progress rectangle coordinates
            self._canvas.coords(
                self._progress_rectangle,
                0,
                0,
                x1,
                height,
            )
            # Update the progress rectangle color
            color = self.get_progress_color()
            
            self._canvas.itemconfig(self._progress_rectangle, fill=color, outline=color)
            self.update_idletasks()
        except tkinter.TclError as e:
            if "invalid command name" in str(e): return
            raise(e)

    def _resize_canvas(self, event):
        width = self.winfo_width()
        height = self.winfo_height()

        if 0 in [width, height]: return
        
        wscale = float(event.width)/width
        hscale = float(event.height)/height

        self._canvas.config(width=width, height=height, bg=self.get_background_color())

        # This prevents complaining
        if 0 in [wscale, hscale]: return
        self._canvas.scale("all",0,0,wscale,hscale)

        # Make sure the text is positioned in the center
        center = (width / 2 - 2*self.cget('borderwidth'), height / 2 - 2*self.cget('borderwidth'))
        self._canvas.coords(
            self._text,
            center,
        )
