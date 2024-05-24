import functools

def api(f):
    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
