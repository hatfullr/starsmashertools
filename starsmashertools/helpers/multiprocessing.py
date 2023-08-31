import multiprocessing
import traceback
import sys
import starsmashertools.helpers.stacktrace

# This special class raises the correct exception in the main
# thread when it hits an exception itself.

class Process(multiprocessing.Process, object):
    def __init__(self, *args, **kwargs):
        super(Process, self).__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self, *args, **kwargs):
        try:
            super(Process, self).run(*args, **kwargs)
            self._cconn.send(None)
        except BaseException as e:
            tb = starsmashertools.helpers.stacktrace.format_exc(exception=e, for_raise = True)
            #tb = traceback.format_exc()
            #tb = tb.replace(type(e).__name__+": "+str(e), '').strip()
            self._cconn.send((e, tb))

    def is_alive(self, *args, **kwargs):
        # Whenever the user checks if this process is alive, we check
        # if we have encountered an exception. If so, we raise that
        # exception.
        ret = super(Process, self).is_alive(*args, **kwargs)
        if self.exception is not None: self.raise_exception()
        return ret
    
    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

    def raise_exception(self):
        exc = self.exception
        if exc is None:
            raise Exception("There is no exception to raise")
        exception, traceback = exc

        if isinstance(exception, SystemExit):
            quit()
        
        print(traceback)
        #sys.tracebacklimit = -1
        raise exception
        
