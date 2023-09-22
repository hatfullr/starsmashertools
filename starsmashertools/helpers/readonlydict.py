


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
