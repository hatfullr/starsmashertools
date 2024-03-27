def defer_keyboardinterrupt(message : str = "Deferring KeyboardInterrupt. Raise KeyboardInterrupt again to stop execution"):
    """ For a block of code, finish execution before raising exceptions, unless
    an exception is raised which involves something in the block of code. """
    import signal
    import functools
    
    original_handler = signal.getsignal(signal.SIGINT)
    
    def signal_handler(sig, frame):
        print(message)
        signal.signal(signal.SIGINT, original_handler)
    
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGINT, signal_handler)
            ret = f(*args, **kwargs)
            signal.signal(signal.SIGINT, original_handler)
            return ret
        return wrapper
    return decorator
