import typing
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import matplotlib
import matplotlib.figure
import matplotlib.axes

# Set this to True to allow classes with attribute _is_example = True to be
# included in the search for Figure and Axes classes.
_is_example = False

# Prefer using the Tk backend (seems it's faster)
matplotlib.use('tkagg', force = False)

def get_classes():
    import starsmashertools
    import starsmashertools.mpl.figure
    import starsmashertools.mpl.axes
    import starsmashertools.helpers.path
    import importlib.util
    import sys
    import inspect

    classes = {
        'Figure' : {
            'class' : matplotlib.figure.Figure,
            'modules' : [
                starsmashertools.mpl.figure,
                ['mpl', 'figure', '*.py'],
            ],
        },
        'Axes' : {
            'class' : matplotlib.axes.Axes,
            'modules' : [
                starsmashertools.mpl.axes,
                ['mpl', 'axes', '*.py'],
            ],
        },
    }

    # Prepare the modules for searching
    for key, obj in classes.items():
        actual_modules = []
        for i, module in enumerate(obj['modules']):
            if inspect.ismodule(module):
                actual_modules += [module]
                continue

            # Load from data
            found_modules = []
            for files in starsmashertools.get_data_files(module):
                for f in files:
                    # Load the module in
                    basename = starsmashertools.helpers.path.basename(f)
                    module_name = 'starsmashertools.data.mpl.figure.' + basename[:-len('.py')]
                    spec = importlib.util.spec_from_file_location(module_name, f)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    found_modules += [module]
            actual_modules += found_modules
        obj['modules'] = actual_modules

    # Search the modules for classes
    found = []
    for key, obj in classes.items():
        for module in obj['modules']:
            for name, _class in inspect.getmembers(module, inspect.isclass):
                if not issubclass(_class, obj['class']): continue
                # If the class doesn't have the 'name' attribute, it is ignored.
                if not hasattr(_class, 'name'):
                    starsmashertools.helpers.warnings.warn(
                        "Missing 'name' attribute in class '%s'. Set 'name' in the class to None to remove this warning." % (_class.__qualname__))
                    continue
                if _class.name is None: continue
                # If it is an example class and we're doing an example, ignore.
                if (hasattr(_class, '_is_example') and _class._is_example):
                    if not _is_example: continue
                found += [_class]

    # Ensure unique names
    for _class in found:
        if not issubclass(_class, (
                matplotlib.axes.Axes,
                matplotlib.figure.Figure,
        )): continue
        for __class in found:
            if _class is __class: continue
            if _class.name != __class.name: continue
            raise ValueError("classes '%s' and '%s' have the same name: '%s'. Make sure each Figure and Axes subclass has a 'name' attribute defined." % (_class.__qualname__, __class.__qualname__, _class.name))
    
    # Return all classes indiscriminantly
    return found


def _register_axes(classes : list = []):
    """ Register all our defined Axes classes with Matplotlib. See 
    https://stackoverflow.com/a/48593767 """
    registered = matplotlib.projections.get_projection_names()
    if not classes: classes = get_classes()
    for _class in classes:
        if not issubclass(_class, matplotlib.axes.Axes): continue
        if _class.name not in registered:
            matplotlib.projections.register_projection(_class)


@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def subplots(
        FigureClass : str | type(None) = 'Figure',
        AxesClass : str | type(None) = 'Axes',
        style : str | type(None) = 'starsmashertools',
        subplot_kw : dict | type(None) = None,
        **kwargs
):
    """
    Similar to :func:`matplotlib.pyplot.subplots`, except allows for the 
    use of custom :class:`matplotlib.figure.Figure` and 
    :class:`matplotlib.axes.Axes` classes.

    Other Parameters
    ----------------
    FigureClass : str, Figure class, None, default = 'Figure'
        The kind of :class:`matplotlib.figure.Figure` to create. If a `str` is 
        given, it is an identifier which should match a ``name`` property of one
        of the custom Figure subclasses in :mod:`~.mpl.figure`. If a Figure 
        class is given, it is used. If `None`, the regular 
        :class:`matplotlib.figure.Figure` class is used.

    AxesClass : str, Axes class, None, default = 'Axes'
        The kind of :class:`matplotlib.axes.Axes` to create. If a `str` is 
        given, it is an identifier which should match a ``name`` property of one
        of the custom Axes subclasses in :mod:`~.mpl.figure`. If an Axes class 
        is given, it is used. If `None`, the regular 
        :class:`matplotlib.axes.Axes` class is used. If you provide your own
        class, it must be registered as a projection with Matplotlib first.

    style : str, None, default = 'starsmashertools'
        The name of a Matplotlib stylesheet to use. If `None` then no stylesheet
        is used.

    subplot_kw : dict, None, defualt = None
        The same as keyword ``subplot_kw`` in 
        :func:`matplotlib.pyplot.subplots`, except the key ``'projection'`` will
        always be set to the value of keyword argument ``AxesClass`` if it is 
        not `None`. If this keyword is `None` and ``AxesClass`` is not `None`, 
        then a new ``subplot_kw`` :py:class:`dict` will be created.

    **kwargs
        Other keyword arguments are passed directly to 
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`

    ax : Matplotlib Axes or array of Axes
        The class provided by ``AxesClass`` will be used for each.
    """
    import matplotlib.pyplot as plt
    import starsmashertools.mpl.figure
    import starsmashertools.helpers.string
    # Importing axes.py registers our Axes classes for projections
    import starsmashertools.mpl.axes

    classes = get_classes()
    _register_axes(classes = classes)
    
    if FigureClass is None: FigureClass = matplotlib.figure.Figure
    elif isinstance(FigureClass, str):
        # Check for the string identifier in our custom Figure classes
        for _class in classes:
            if not issubclass(_class, matplotlib.figure.Figure): continue
            if not hasattr(_class, 'name'): continue
            if _class.name != FigureClass: continue
            FigureClass = _class
            break
        else:
            names = []
            for _class in classes:
                if not issubclass(_class, matplotlib.figure.Figure): continue
                if not hasattr(_class, 'name'): continue
                names += [_class.name]
            s = starsmashertools.helpers.string.list_to_string(names, join='or')
            raise ValueError("Unrecognized FigureClass '%s'. The available names are %s" % (FigureClass, s))
    
    # Search for custom Axes
    if isinstance(AxesClass, str):
        if subplot_kw is None: subplot_kw = dict(projection = AxesClass)
        else: subplot_kw['projection'] = AxesClass
    
    if style is not None:
        starsmashertools.mpl.figure.update_style_sheet_directories()
        plt.style.use(style)
    
    fig, ax = plt.subplots(
        FigureClass = FigureClass,
        subplot_kw = subplot_kw,
        **kwargs
    )
    fig.draw(renderer = fig.canvas.get_renderer())
    return fig, ax



