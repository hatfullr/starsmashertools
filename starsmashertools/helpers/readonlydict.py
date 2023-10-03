import copy

class ReadOnlyDict(dict, object):
    def raise_readonly(*args, **kwargs):
        raise Exception("Cannot edit a ReadOnlyDict")

    __setitem__ = raise_readonly
    __delitem__ = raise_readonly
    pop = raise_readonly
    popitem = raise_readonly
    clear = raise_readonly
    update = raise_readonly
    setdefault = raise_readonly

    def __copy__(self):
        self.__setitem__ = super(ReadOnlyDict, self).__setitem__
        ret = self.__class__.__new__(self.__class__)
        ret.__dict__.update(self.__dict__)
        self.__setitem__ = raise_readonly
        return ret
    def __deepcopy__(self, memo):
        self.__setitem__ = super(ReadOnlyDict, self).__setitem__
        ret = self.__class__.__new__(self.__class__)
        memo[id(self)] = ret
        for k, v in self.__dict__.items():
            setattr(ret, k, copy.deepcopy(v, memo))
        self.__setitem__ = raise_readonly
        return ret
