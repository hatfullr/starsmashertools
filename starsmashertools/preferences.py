"""
This is where code preferences are managed.
"""
import inspect, functools
import copy
from starsmashertools import SOURCE_DIRECTORY
import os

# Detect if this file is imported during the execution of a code within the
# starsmashertools source directory (except the bin directory)
frame = inspect.currentframe()
while frame.f_back is not None:
    frame = frame.f_back
current_directory = os.path.realpath(os.path.dirname(inspect.getsourcefile(frame)))
_execution_in_source = False
if current_directory.startswith(os.path.realpath(SOURCE_DIRECTORY)):
    _execution_in_source = not current_directory.startswith(os.path.join(os.path.realpath(SOURCE_DIRECTORY), 'bin'))

del current_directory
del frame
del SOURCE_DIRECTORY




class NoDefaultPreference(object): pass

# An object which can be used in function argument annotations for specifying
# that whatever value is in the preferences should be used.
class Pref(object):
    """
    This class can be used in function argument annotations for specifying
    that the preferences should be used. You must give the name of the 
    preference.

    Examples
    --------

    In a fake example module ``starsmashertools.example``, we define::

         import starsmashertools.preferences
         from starsmashertools.preferences import Pref
         
         @starsmashertools.preferences.use
         class A:
              def __init__(self, val = Pref('val')):
                  print(val)
         
         A()
    
    Let the preferences files contain an identifier ``example.A.val`` with a
    value of ``'foo'``. The above code will print ``'foo'`` to the terminal.
    
    """
    
    def __new__(cls, name, default = NoDefaultPreference()):
        if _execution_in_source: return super().__new__(cls)
        
        # First try just the given name, in case it's a full preference ID
        try:
            return Preferences.get_state(name)
        except: pass
        
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        finfo = inspect.getframeinfo(frame)
        
        if finfo.function == '<module>': # Function defined in outermost scope in module
            return super().__new__(cls)
        
        qualname = '.'.join([module.__name__, finfo.function, name])
        try:
            return Preferences.get_static(qualname)
        except PreferenceNotFoundError as e:
            if not isinstance(default, NoDefaultPreference): return default
            raise PreferenceNotFoundError("'%s' in module '%s'" % (qualname, module.__name__)) from e
        
    def __init__(self, name, default = NoDefaultPreference()):
        self.name = name
        self.default = default

def _fix_function(_obj, preferences = None):
    """ Replace any Pref objects found in the default values with values from
    the preferences files. """

    if _execution_in_source: return
    
    if preferences is None: preferences = Preferences(_obj)
    
    if _obj.__defaults__:
        defaults = list(_obj.__defaults__)
        for i, val in enumerate(defaults):
            if isinstance(val, Pref):
                defaults[i] = preferences.get(val.name)
        _obj.__defaults__ = tuple(defaults)
    if _obj.__kwdefaults__:
        for key, val in _obj.__kwdefaults__.items():
            if isinstance(val, Pref):
                _obj.__kwdefaults__[key] = preferences.get(val.name)
        
def use(_obj):
    """ A class decorator. Put this decorator on a class to allow obtaining
    preferences anywhere inside that class. This allows for 
    ``self.preferences`` to work. This also works on functions. """
    
    if inspect.isclass(_obj):
        _obj.preferences = Preferences(_obj)
        for attr_name in dir(_obj):
            attr = getattr(_obj, attr_name)
            if not inspect.isfunction(attr): continue
            _fix_function(attr, preferences = _obj.preferences)
    else:
        _fix_function(_obj)
        
        @functools.wraps(_obj)
        def wrapper(*args, **kwargs): return _obj(*args, **kwargs)
        return wrapper
    
    return _obj

class PreferenceNotFoundError(Exception, object): pass
class PreferenceExcludedError(Exception, object): pass

class Preferences(object):
    """
    An object to handle user and default preferences with respect to a given
    class that was decorated by :func:`~.use`.

    The user's preferences file contains a :py:class:`dict` called ``user``, 
    which is a branching tree of dictionaries. The leaves of the tree can have 
    types :py:class:`int`, :py:class:`float`, :py:class:`complex`, 
    :py:class:`bool`, :py:class:`str`, :py:class:`bytes`, or ``None``.

    To obtain a preference, first the ``user`` :py:class:`dict` is traversed 
    using keys from a string identifier (see :meth:`~.get`). Each key is 
    followed in order. At each step, the key is checked against the ``exclude``
    :py:func:`dict` in the user's preferences and a 
    :class:`~.PreferenceExcludedError` is raised if it matches. After the last 
    key in the identifier has been found, the resulting object is returned as 
    the requested preference. If the preference wasn't found in ``user`` and it
    a :class:`~.PreferenceExcludedError` was not raised, then the ``default`` 
    :py:class:`dict` is checked. If the identifier still isn't found, a 
    :class:`~.PreferenceNotFoundError` is raised.

    Each leaf of the ``exclude`` :py:class:`dict` is of type :py:class:`list`.
    Whenever a key in the identifier is checked, the ``exclude`` dict is also
    checked for the corresponding key. If the current key being checked is
    included in the leaf, a :class:`~.PreferenceExcludedError` is raised.

    Examples
    --------
    Suppose the preference dictionaries are defined as follows:
    
        # data/user/preferences.py
        prefs = {
            'module' : {
                'Class' : {
                    'foo' : 'bar',
                    'item' : None,
                    'mydict' : {
                        '2' : {
                            '2a' : 4,
                        },
                    },
                },
                'Class2' : {
                    'obj' : {
                        'i2' : 'c',
                    },
                },
            },
        }

        exclude = [
            'module.Class.item',
            'module.Class2.obj.i2',
        ]

        # data/defaults/preferences.py
        prefs = {
            'module' : {
                'Class' : {
                    'foo' : 0,
                    'item' : 100.,
                    'verbose' : True,
                    'mydict' : {
                        '1' : {
                            '1a' : 0,
                            '1b' : 1,
                        },
                        '2' : {
                            '2a' : 2,
                            '2b' : 6,
                        },
                    },
                },
                'Class2' : {
                    'obj' : {
                        'i1' : 'a',
                        'i2' : 'b',
                    },
                },
            },
        }
    
        
    Let there exist a class ``module.Class`` (such as in a file called 
    "module.py"):
        
        import starsmashertools.preferences
        
        @starsmashertools.preferences.use
        class Class:
            def __init__(self):
                Class.preferences.get('foo') # 'bar'
                Class.preferences.get('item') # PreferenceExcludedError
                Class.preferences.get('verbose') # True
                Class.preferences.get('a') # PreferenceNotFoundError
                Class.preferences.get('mydict.1.1a') # 0
                Class.preferences.get('mydict.2.2a') # 4
                Class.preferences.get('mydict.2.2b') # 6

        @starsmashertools.preferences.use
        class Class2:
            def __init__(self):
                Class2.preferences.get('obj') # {'i1': 'a'}
                Class2.preferences.get('a') # PreferenceNotFoundError
                Class2.preferences.get('obj.i2') # PreferenceExcludedError
    """
    _user = None
    _exclude = None
    _default = None
    
    def __init__(self, _obj):
        self._obj = _obj

    @property
    def user(self):
        if Preferences._user is None:
            Preferences._user = Preferences._get_user_dict()
        return copy.deepcopy(Preferences._user)

    @property
    def default(self):
        if Preferences._default is None:
            Preferences._default = Preferences._get_default_dict()
        return copy.deepcopy(Preferences._default)

    @property
    def exclude(self):
        if Preferences._exclude is None:
            Preferences._exclude = Preferences._get_exclude()
        return copy.deepcopy(Preferences._exclude)
    
    @staticmethod
    def _get_user_dict():
        import sys, starsmashertools
        import starsmashertools.helpers.path
        import importlib.util
        prefs = {}
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if user:
            spec = importlib.util.spec_from_file_location('preferences', user[0])
            module = importlib.util.module_from_spec(spec)
            #sys.modules['__mypreferences__'] = module
            spec.loader.exec_module(module)
            if hasattr(module, 'prefs'): prefs = module.prefs
            #del sys.modules['__mypreferences__']
        return prefs
    
    @staticmethod
    def _get_exclude():
        import sys, starsmashertools
        import starsmashertools.helpers.path
        import importlib.util
        exclude = []
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if user:
            spec = importlib.util.spec_from_file_location('preferences', user[0])
            module = importlib.util.module_from_spec(spec)
            #sys.modules['__mypreferences__'] = module
            spec.loader.exec_module(module)
            if hasattr(module, 'exclude'): exclude = module.exclude
            #del sys.modules['__mypreferences__']
        return exclude
    
    @staticmethod
    def _get_default_dict():
        import sys, starsmashertools
        import starsmashertools.helpers.path
        import importlib.util
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if not default:
            raise FileNotFoundError("Failed to find 'starsmashertools/data/defaults/preferences.py'. Your installation might be corrupt.")
        spec = importlib.util.spec_from_file_location('preferences', default[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prefs = module.prefs
        return prefs

    @staticmethod
    def _traverse_static(dictionary : dict, identifier : str):
        current = dictionary
        towalk = identifier.split('.')
        if not towalk: raise PreferenceNotFoundError(identifier)
        if towalk[0] == 'starsmashertools': towalk = towalk[1:]
        for key in towalk:
            if key in current.keys(): current = current[key]
            else: raise PreferenceNotFoundError(identifier)
        return current
    
    def _traverse(self, dictionary : dict, identifier : str):
        def do(obj):
            return Preferences._traverse_static(
                dictionary,
                '.'.join([obj.__module__, obj.__qualname__, identifier]),
            )

        if not inspect.isclass(self._obj): return do(self._obj)
        else:
            try:
                return do(self._obj)
            except PreferenceNotFoundError:
                # Search the classes which this object inherits from
                for inherited in self._obj.__mro__:
                    if inherited == self._obj: continue
                    try:
                        return do(inherited)
                    except PreferenceNotFoundError: continue
                raise
    
    def get_user(self, identifier : str):
        """
        Traverse the ``user`` :py:class:`dict` for the given identifier.
        
        Raises a :class:`~.PreferenceMissingError` if no the identifier is not
        found.
        
        Raises a :class:`~.PreferenceExcludedError` if the identifier is in the
        ``exclude`` :py:class:`dict`.

        Parameters
        ----------
        identifier : str
            See :meth:`~.get`.

        Returns
        -------
        :py:class:`dict`, :py:class:`int`, :py:class:`float`, :py:class:`complex`, :py:class:`bool`, :py:class:`str`, :py:class:`bytes`, or ``None``
            The value of the preference requested.
        
        See Also
        --------
        :meth:`~.get`, :meth:`~.get_default`
        """
        if identifier in self.exclude: raise PreferenceExcludedError
        return self._traverse(self.user, identifier)
    
    def get_default(self, identifier : str):
        """
        Traverse the ``default`` :py:class:`dict` for the given identifier.
        
        Raises a :class:`~.PreferenceMissingError` if no preference is found.

        Parameters
        ----------
        identifier : str
            See :meth:`~.get`.

        Returns
        -------
        dict, int, float, complex, bool, str, bytes, or None
            The value of the preference requested.

        See Also
        --------
        :meth:`~.get`, :meth:`~.get_user`
        """
        return self._traverse(self.default, identifier)
    
    def get(self, identifier : str):
        """
        Obtain the preference given by ``identifier``, either in the user's 
        preferences or in the default preferences.

        Raises a :class:`~.PreferenceMissingError` if the preference wasn't
        found in either the ``user`` :py:class:`dict` or the ``default`` 
        :py:class:`dict`.

        Parameters
        ----------
        identifier : str
            A string delimited by '.', which describes the keys on which to
            traverse the preference dictionaries. Appears as 
            ``'key1.key2.key3'``, etc. Two keys are automatically inserted: 
            ``'module.class.key1.key2.key3'``, where ``'module'`` is the name of
            the module to which ``'class'`` belongs. The class is the same as 
            that which was decorated with the :func:`~.use` function.
        
        Returns
        -------
        dict, int, float, complex, bool, str, bytes, or None
            The value of the preference requested. 
        """
        try:
            return self.get_user(identifier)
        except PreferenceNotFoundError: pass
        return self.get_default(identifier)

    @staticmethod
    def get_static(identifier : str):
        """
        Search the preferences for the given full identifier, which must appear
        as, e.g., ``starsmashertools.lib.simulation.Simulation``, where the
        ``Simulation`` class in module ``starsmashertools.lib.simulation`` has
        the ``starsmashertools.preferences.use`` decorator.
        """

        if Preferences._user is None:
            Preferences._user = Preferences._get_user_dict()
        user = copy.deepcopy(Preferences._user)

        if Preferences._exclude is None:
            Preferences._exclude = Preferences._get_exclude()
        exclude = copy.deepcopy(Preferences._exclude)

        if Preferences._default is None:
            Preferences._default = Preferences._get_default_dict()
        default = copy.deepcopy(Preferences._default)

        if exclude:
            if identifier in exclude: raise PreferenceExcludedError

        try:
            return Preferences._traverse_static(user, identifier)
        except PreferenceNotFoundError:
            return Preferences._traverse_static(default, identifier)
