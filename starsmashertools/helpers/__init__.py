def defer_keyboardinterrupt(message : str = "Deferring KeyboardInterrupt. Raise KeyboardInterrupt again to stop execution"):
    """ For a block of code, finish execution before raising exceptions, unless
    an exception is raised which involves something in the block of code. """
    import signal
    import functools
    import starsmashertools.helpers.asynchronous
    
    original_handler = signal.getsignal(signal.SIGINT)
    
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):

            if not starsmashertools.helpers.asynchronous.is_main_process():
                return f(*args, **kwargs)
            
            previous_handler = original_handler
            if hasattr(f, '_signal_handler'):
                previous_handler = f._signal_handler
            
            def signal_handler(sig, frame):
                print(message)
                signal.signal(signal.SIGINT, original_handler)
            
            f._signal_handler = signal_handler
            signal.signal(signal.SIGINT, signal_handler)
            
            ret = f(*args, **kwargs)

            f._signal_handler = previous_handler
            signal.signal(signal.SIGINT, previous_handler)
            
            return ret
        return wrapper
    return decorator
