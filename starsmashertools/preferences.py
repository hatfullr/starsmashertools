"""
This is where code preferences are managed.
"""
import inspect, functools
import copy

def use(_obj):
    """ A class decorator. Put this decorator on a class to allow obtaining
    preferences anywhere inside that class. This allows for 
    ``self.preferences`` to work. This also works on functions. """
    
    _obj.preferences = Preferences(_obj)
    
    if not inspect.isclass(_obj):
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
        import sys, os, starsmashertools
        import starsmashertools.helpers.path
        prefs = {}
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if user:
            orig_path = copy.deepcopy(sys.path)
            sys.path.insert(1, starsmashertools.helpers.path.dirname(user[0]))
            from preferences import prefs
            sys.path = orig_path
        return prefs
    
    @staticmethod
    def _get_exclude():
        import sys, os, starsmashertools
        import starsmashertools.helpers.path
        exclude = []
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if user:
            orig_path = copy.deepcopy(sys.path)
            sys.path.insert(1, starsmashertools.helpers.path.dirname(user[0]))
            try:
                from preferences import exclude
            except ImportError as e: pass
            sys.path = orig_path
        return exclude
    
    @staticmethod
    def _get_default_dict():
        import sys, os, starsmashertools
        import starsmashertools.helpers.path
        prefs = {}
        default, user = starsmashertools.get_data_files(['preferences.py'])
        if not default:
            raise FileNotFoundError("Failed to find 'starsmashertools/data/defaults/preferences.py'. Your installation might be corrupt.")
        orig_path = copy.deepcopy(sys.path)
        sys.path.insert(1, starsmashertools.helpers.path.dirname(default[0]))
        from preferences import prefs
        sys.path = orig_path
        return prefs
    
    def _traverse(self, dictionary : dict, identifier : str):
        def do(obj):
            _identifier = '{module:s}.{qualname:s}.{identifier:s}'.format(
                module = obj.__module__.replace('starsmashertools.',''),
                qualname = obj.__qualname__,
                identifier = identifier,
            )
            current = dictionary
            for key in _identifier.split('.'):
                if key in current.keys(): current = current[key]
                else: raise PreferenceNotFoundError(_identifier)
            return current

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
