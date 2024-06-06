import collections.abc
import copy
import gzip
import pathlib
import shutil
import pickle
import tempfile
import os

def is_path_from_stems(path, stems):
    """ Check all the given stems to see if the given path starts with one of 
    them. """
    if isinstance(path, str): return path in stems
    if len(path) == 0: return False
    
    for stem in stems:
        if isinstance(stem, str):
            if path[0] == stem: return True
            continue
        if path[:len(stem)] == stem: return True
    return False

class nested_dict_keys(collections.abc.KeysView):
    def __init__(self, mapping, stems = ()):
        if not isinstance(mapping, NestedDict):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of types NestedDict, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._stems = stems

    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        # If the path is in one of the stems, yield
        if not stems or is_path_from_stems(path, stems):
            if len(path) == 1: yield path[0]
            elif len(path) > 1: yield path
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_keys._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_keys._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
    
    @property
    def _mapping(self): return list(self._get_mapping_gen(self.mapping, self._stems))
    def __reversed__(self): return reversed(self._mapping)
    def __contains__(self, key):
        if isinstance(key, tuple) and len(key) == 1:
            return super(nested_dict_keys, self).__contains__(key[0])
        return super(nested_dict_keys, self).__contains__(key)

# This is what's done in the Python source code, so we'll do it too
collections.abc.KeysView.register(nested_dict_keys)


class nested_dict_branches(nested_dict_keys):
    """
    This is similar to :class:`~.nested_dict_keys`\, but only the keys which 
    reference non-dict values are included. These are the "branches" of the 
    nested dict "tree". The values of these keys are the "leaves".
    """
    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_branches._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_branches._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        else:
            # If the path is in one of the stems, yield
            if not stems or is_path_from_stems(path, stems):
                if len(path) == 1: yield path[0]
                elif len(path) > 1: yield path
    @property
    def _mapping(self): return list(self._get_mapping_gen(self.mapping, self._stems))
collections.abc.KeysView.register(nested_dict_branches)

class nested_dict_stems(nested_dict_keys):
    """
    Similar to :class:`~.nested_dict_keys`\, but the keys included are either 
    those which don't point to a leaf, or those which do point to a leaf but
    have a path length of 1 (at the root of the tree).
    """
    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        branches = list(nested_dict_branches._get_mapping_gen(obj, stems, path = path))
        for key in nested_dict_keys._get_mapping_gen(obj, stems, path = path):
            if key in branches: continue
            yield key
collections.abc.KeysView.register(nested_dict_stems)


class nested_dict_values(collections.abc.ValuesView):
    def __init__(self, mapping, stems = ()):
        if not isinstance(mapping, NestedDict):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of types NestedDict, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._stems = stems

    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        if len(path) != 0:
            if not stems or is_path_from_stems(path, stems): yield obj
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_values._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_values._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
                
    @property
    def _mapping(self): return list(self._get_mapping_gen(self.mapping, self._stems))
    def __iter__(self): yield from self._mapping
    def __reversed__(self): return reversed(self._mapping)
    def __contains__(self, value): return value in self._mapping
collections.abc.ValuesView.register(nested_dict_values)

class nested_dict_leaves(nested_dict_values):
    """
    The "leaves" are the values at the end of each "branch" in the nested dict
    "tree". Leaves are never of type :py:class:`dict`\.
    """
    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_leaves._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_leaves._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif len(path) != 0:
            if not stems or is_path_from_stems(path, stems): yield obj
collections.abc.ValuesView.register(nested_dict_leaves)


class nested_dict_items(collections.abc.ItemsView):
    def __init__(self, mapping, stems = ()):
        if not isinstance(mapping, NestedDict):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of types NestedDict, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._stems = stems
    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        if not stems or is_path_from_stems(path, stems):
            if len(path) == 1: yield path[0], obj
            elif len(path) > 1: yield path, obj
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_items._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_items._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )

    @property
    def _mapping(self):
        return [(k,v) for k,v in self._get_mapping_gen(self.mapping, self._stems)]
    def __iter__(self): yield from self._mapping
    def __reversed__(self): return reversed(self._mapping)
    def __contains__(self, item):
        key, value = item
        try: v = self.mapping[key]
        except KeyError: return False
        else: return v is value or v == value
collections.abc.ItemsView.register(nested_dict_items)

class nested_dict_flowers(nested_dict_items):
    @staticmethod
    def _get_mapping_gen(obj, stems, path = ()):
        if isinstance(obj, NestedDict):
            for key, val in super(NestedDict, obj).items():
                yield from nested_dict_flowers._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        elif isinstance(obj, dict):
            for key, val in obj.items():
                yield from nested_dict_flowers._get_mapping_gen(
                    val, stems, path = tuple(list(path) + [key]),
                )
        else:
            if not stems or is_path_from_stems(path, stems):
                if len(path) == 1: yield path[0], obj
                elif len(path) > 1: yield path, obj
collections.abc.ItemsView.register(nested_dict_flowers)




class NestedDict(dict, object):
    """
    Nested dictionaries quickly become hard to handle in Python. This class is
    intended to make it easier. It works like a regular :py:class:`dict`\, but
    keys can be specified as a list of nest levels. For example, consider a
    :class:`~.NestedDict` with a structure of
    
    .. code-block:: python
    
        nested = NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

    To access the key '1aii', one can do ``nested['1','1a','1aii']``\. Setting
    values works similarly. The builtin standard :py:class:`dict` methods are
    overridden to restore the expected functionality. For example,
    ``nested.keys()`` would return ``nested_dict_keys(['1', ['1', '1a'], ['1', '1a', '1ai'], ['1', '1a', '1aii'], ['1', '1b'], ['1', '1b', '1bi'], '2'])``\.
    Note that this class uses custom dictionary View objects.

    Use :meth:`~.branches` to get all the unique keys, regardless of their depth
    in the nested dict, use :meth:`~.leaves` to get the values of the unique
    keys, and use :meth:`~.flowers` to get the combination of branches and
    leaves (analogous to ``dict.items()``\). The values returned by 
    :meth:`~.branches` is not the same as that returned by :meth:`~.keys`\, and
    likewise for :meth:`~.leaves` and :meth:`~.flowers`\.

    To convert to a regular dictionary, use :meth:`~.to_dict`\.
    """
    def __init__(self, iterable = None, **kwargs):
        super(NestedDict, self).__init__({}, **kwargs)
        if iterable is not None:
            if isinstance(iterable, NestedDict):
                for key, val in iterable.items():
                    self[key] = val
            else:
                for key,val in dict(iterable, **kwargs).items():
                    if isinstance(key, tuple):
                        raise KeyError("NestedDict cannot be instantiated from a dictionary which has a tuple in its keys: %s" % str(key))
                    self[key] = val
        
    
    def __getitem__(self, key):
        #if isinstance(key, list): key = tuple(key)
        try: return super(NestedDict, self).__getitem__(key)
        except KeyError: pass
        # Try accessing using our fancy methods
        for k, val in self.items():
            if k == key: return val
        raise KeyError(str(key))

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) > 1:
            current = self
            for k in key[:-1]:
                if k not in current.keys(): current[k] = {}
                current = current[k]
            current[key[-1]] = value
        else:
            return super(NestedDict, self).__setitem__(key, value)
    
    def __delitem__(self, key):
        try: return super(NestedDict, self).__delitem__(key)
        except KeyError: pass
        current = self
        for level in key[:-1]:
            current = current[level]
        del current[key[-1]]

    def __len__(self): return len(self.items())
    def __contains__(self, key): return key in self.keys()
    
    def __iter__(self): return iter(self.branches())
    
    def __reversed__(self): return reversed(self.keys())
    def __or__(self, other):
        cpy = copy.deepcopy(self)
        if isinstance(other, NestedDict): cpy.update(other)
        else:
            for key, val in other.items(): cpy[key] = val
        return cpy
    def __ror__(self, other):
        if not isinstance(other, dict): raise TypeError
        for key, val in self.flowers():
            current = other
            for k in key[:-1]:
                if k not in current.keys(): current[k] = {}
                current = current[k]
            current[key[-1]] = val
        return other
    
    def __ior__(self, other):
        self.update(other)
        return self

    def __eq__(self, other):
        if isinstance(other, NestedDict):
            branches = self.branches()
            other_branches = other.branches()
            if len(branches) != len(other_branches): return False
            for branch in branches:
                if branch not in other_branches: return False
                if self[branch] != other[branch]: return False
            for branch in other_branches:
                if branch not in branches: return False
                if self[branch] != other[branch]: return False
        elif isinstance(other, dict):
            for key, val in self.flowers():
                current = other
                for k in key[:-1]:
                    current = current[k]
                if current[key[-1]] != val: return False
        else:
            raise TypeError(str(other))
        return True

    def __ne__(self, other): return not (self == other)
    
    def branches(self, *args, **kwargs):
        """ Keys in the nested dictionary which point to "leaves" in the nested
        dict "tree". Each "leaf" is not a :py:class:`dict`\, by definition. """
        return nested_dict_branches(self, *args, **kwargs)
    def stems(self, *args, **kwargs):
        """ Keys in the nested dictionary which don't point to "leaves" in the
        nested dict "tree", but rather point to further nesting levels in the
        tree. """
        return nested_dict_stems(self, *args, **kwargs)
    def leaves(self, *args, **kwargs):
        """ Non-dict values in the nested dict "tree". Each of these values
        correspond to a key, given by :meth:`~.branches`\. """
        return nested_dict_leaves(self, *args, **kwargs)

    def flowers(self, *args, **kwargs):
        """ A combination of branches and leaves, in the same way that keys and
        values are combined to form "items" in a :py:class:`dict`\. """
        return nested_dict_flowers(self, *args, **kwargs)

    def keys(self,*args,**kwargs): return nested_dict_keys(self,*args,**kwargs)
    def values(self,*args,**kwargs): return nested_dict_values(self,*args,**kwargs)
    def items(self,*args,**kwargs): return nested_dict_items(self,*args,**kwargs)

    @classmethod
    def fromkeys(cls, iterable, value = None):
        self = cls()
        # Set 'deep' items first
        iterobj = list(iterable)
        order = [len(i) if isinstance(i, tuple) else 0 for i in iter(iterobj)]
        disallowed_keys = []
        for key in [x for _,x in sorted(zip(order,iterobj),key=lambda pair:pair[0])][::-1]:
            # If a key will be set which contains a nesting, throw an error
            if isinstance(key, tuple):
                for k in key:
                    if k in disallowed_keys: raise KeyError("Cannot set key because it would overwrite data specified by other keys that have already been set: %s" % str(key))
            elif key in disallowed_keys: raise KeyError("Cannot set key because it would overwrite data specified by other keys that have already been set: %s" % str(key))
            
            self[key] = value
            if isinstance(key, tuple):
                disallowed_keys += key
            else: disallowed_keys += [key]
        return self

    def get(self, key, default = None):
        if key not in self: return default
        return self[key]
    
    def pop(self, key, default = None):
        ret = self[key]
        del self[key]
        return ret

    def popitem(self):
        if not self: raise KeyError
        ret = list(self.items())[-1]
        del self[ret[0]]
        return ret

    def setdefault(self, key, default = None):
        if key in self: return self[key]
        self[key] = default
        return default

    def update(self, other, **kwargs):
        if isinstance(other, NestedDict):
            for key, val in other.flowers(): self[key] = val

        elif isinstance(other, dict):
            for key, val in other.items():
                if isinstance(val, dict):
                    # If the dictionary is nested
                    self.update(NestedDict(other), **kwargs)
                    return
            # If the dictionary isn't nested
            for key, val in other.items(): self[key] = val
        else:
            # key/value pairs are given
            for key, val in other: self[key] = val
        
        for key, val in kwargs.items():
            self[key] = val

    def to_dict(self):
        """ Return a copy of this NestedDict as a regular Python 
        :py:class:`dict`\. """
        ret = {}
        for branch, leaf in self.flowers():
            current = ret
            if isinstance(branch, tuple): # A nested branch
                for key in branch[:-1]:
                    if key not in current.keys(): current[key] = {}
                    current = current[key]
                current[branch[-1]] = leaf
            else: # A regular dict key
                current[branch] = leaf
        return ret

    def get_stems(self, branch):
        """ Given a branch, return its stems. """
        if branch not in self.branches(): raise KeyError(branch)
        if isinstance(branch, str): branch = (branch,)
        for stem in self.stems():
            if isinstance(stem, tuple):
                if branch[:len(stem)] != stem: continue
            else:
                if branch[0] != stem: continue
            yield stem

    @staticmethod
    def is_child_branch(
            child : str | tuple,
            parent : str | tuple,
    ):
        """ Returns `True` if the given ``child`` branch is contained within the
        given ``parent`` branch, and `False` otherwise. For example, if
        ``child = ('root', 'item', 'leaf')`` and ``parent = ('root',)``\, then
        the result is `True`\. However, if ``parent = ('root', 'other')``\, then
        the result is `False`\. 

        If ``child`` and ``parent`` refer to identical branches, or ``child`` 
        is a shorter path than ``parent``, returns `False`\.
        """
        if isinstance(child, str): child = (child,)
        if isinstance(parent, str): parent = (parent,)

        if len(child) <= len(parent): return False

        for c, p in zip(child, parent):
            if c != p: return False

        return True
        
    def save(
            self,
            filename : str | pathlib.Path,
            branches : list | tuple | type(None) = None,
    ):
        """
        Save the :class:`~.NestedDict` to a file. If the file already exists,
        the object is saved to a temporary file which is then renamed to the
        given ``filename``\. Otherwise, the file is created with ``filename``\.
        
        The information is saved using the ``pickle`` library. Each branch and
        leaf is saved using ``pickle.dump``\, first on the branch, and then on
        the leaf.

        The file is automatically compressed using ``gzip``\.

        Parameters
        ----------
        filename : str, :class:`pathlib.Path`
            The path to the file to create or overwrite.
        
        branches : list, tuple, None, default = None
            Branches to save to the file. If `None`\, all the branches are 
            saved.

        See Also
        --------
        :meth:`~.load`
        """
        if branches is None: branches = self.branches()

        if os.path.exists(filename): # Safe overwriting
            try:
                tname = None
                with tempfile.NamedTemporaryFile(delete=False) as output:
                    tname = output.name
                    _input = gzip.GzipFile(mode = 'wb', fileobj = output)
                    for branch in branches:
                        pickle.dump(branch, _input)
                        pickle.dump(self[branch], _input)
            except:
                if tname is not None and os.path.exists(tname): os.remove(tname)
                raise
            else: # No errors
                os.rename(tname, filename)
        else:
            try:
                with gzip.open(filename, 'wb') as f:
                    for branch in branches:
                        pickle.dump(branch, f)
                        pickle.dump(self[branch], f)
            except:
                if os.path.exists(filename): os.remove(filename)
                raise

    @staticmethod
    def load(
            filename : str | pathlib.Path,
            branches : list | tuple | type(None) = None,
    ):
        """
        Load a file which was saved by :meth:`~.save`\.

        Parameters
        ----------
        filename : str, :class:`pathlib.Path`
            The path to the file to load.
        
        branches : list, tuple, None, default = None
            Branches to load from the file. If `None`\, all the branches are 
            loaded.
        
        See Also
        --------
        :meth:`~.save`
        """
        ret = NestedDict()
        with gzip.open(filename, 'rb') as f:
            branch = pickle.load(f)
            if branches is None or branch in branches:
                ret[branch] = pickle.load(f)
            else: pickle.load(f) # Skip it
        return ret
