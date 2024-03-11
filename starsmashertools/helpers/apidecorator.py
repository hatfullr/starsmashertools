import functools

def api(verbose : bool = True, progress : bool = False, **kw):
    import starsmashertools.helpers.string
    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            import starsmashertools.helpers.string
            import starsmashertools.helpers.asynchronous
            import sys
            import inspect

            if (not verbose or
                # Don't show loading messages for any subprocesses
                not starsmashertools.helpers.asynchronous.is_main_process() or
                # Output isn't going to terminal
                not sys.stdout.isatty()):
                return f(*args, **kwargs)
            
            message = f.__name__
            
            if progress:
                with starsmashertools.helpers.string.progress_message(
                        message = message, **kw,
                ) as progress:
                    f.progress = progress
                    return f(*args, **kwargs)
            else:
                with starsmashertools.helpers.string.loading_message(
                        message = message, **kw
                ):
                    return f(*args, **kwargs)
        
        return wrapper
    return decorator
