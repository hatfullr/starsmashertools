import multiprocessing
import multiprocessing.queues
import typing

max_processes = multiprocessing.cpu_count()
semaphore = multiprocessing.Semaphore(max_processes)

def is_main_process(): return multiprocessing.parent_process() is None

class Process(multiprocessing.Process, object):
    def __init__(self, *args, target = None, **kwargs):
        if not is_main_process():
            raise Exception("Spawning processes is only allowed on the main process.")

        # Don't allow spawning more processes than the total number of CPUs.
        self._original_target = target
        with semaphore:
            super(Process, self).__init__(*args, target = self._execute_target, **kwargs)

    def _execute_target(self, *args, **kwargs):
        if self._original_target is not None:
            self._original_target(*args, **kwargs)
        
class ParallelFunction(object):
    """
    This object can be used as a shortcut for writing parallel processes. You
    supply a target function which will be run on many individual processes. The
    supplied arguments are 
    """
    
    def __init__(
            self,
            target : typing.Callable,
            args : tuple | list | type(None) = None,
            kwargs : tuple | list | type(None) = None,
            nprocs : int | type(None) = None,
            start : bool = True,
            **kw
    ):
        """
        Initializer. If there are no available processes (because too many have
        already been spawned), whenever processes become available we create 
        processes up to a number ``nprocs``, which must be smaller than
        ``multiprocessing.cpu_count()``.
        
        Parameters
        ----------
        target : callable
            The function to be executed in parallel.

        args : list, tuple, None, default = None
            A list of positional arguments to pass to ``target``. Each element
            should be a list of positional arguments. The number of calls to
            ``target`` is equal to the number of elements in this list. If
            `None`, no arguments will be passed to ``target``.

        kwargs : list, tuple, None, default = None
            A list of keyword arguments to pass to ``target``. Each element
            should be a dictionary of keyword arguments. If ``args`` is not 
            `None` and this list has fewer elements than ``args``, then ``{}`` 
            will be appended until it is the same length as ``args``. 
        

        Other Parameters
        ----------------
        nprocs : int, None, default = None
            The number of processes to use. If `None`, the number of processes
            will be the number of available CPU cores as given by
            ``multiprocessing.cpu_count()``.

        start : bool, default = True
            Automatically start the processes after initialization without
            blocking. To start with blocking, specify `False` and then call
            :func:`~.start` in your code.

        **kw
            Other keyword arguments are passed directly to
            ``multiprocessing.Process`` when creating each individual process.
            By default, keyword 'daemon' is set to `True`.
        """
        if nprocs is None: nprocs = max_processes
        self._nprocs = nprocs
        
        # Populate the input queue
        if args is None and kwargs is None:
            # Just call the function one time per process
            args = [()] * nprocs
            kwargs = [{}] * nprocs
        elif args is None and kwargs is not None:
            args = [()] * len(kwargs)
        elif args is not None and kwargs is None:
            kwargs = [{}] * len(args)
        else: # args and kwargs are not None
            if len(kwargs) > len(args):
                raise Exception("When 'args' is not None and 'kwargs' is not None, 'kwargs' cannot be longer than 'args': %d > %d" % (len(kwargs), len(args)))
            if len(kwargs) < len(args):
                kwargs += [{}] * (len(args) - len(kwargs))

        manager = multiprocessing.Manager()
        self._input_queue = manager.Queue()
        self._output_queue = manager.Queue()
        self._error_queue = manager.Queue()

        self._error_queue_lock = manager.Lock()

        for i, (a, k) in enumerate(zip(args, kwargs)):
            self._input_queue.put([i, a, k])

        self._expected_outputs = self._input_queue.qsize()

        self._target = target
        self._processes = []
        for i in range(nprocs):
            self._processes += [Process(
                target = self.func,
                args = (
                    self._input_queue,
                    self._output_queue,
                    self._error_queue,
                    self._error_queue_lock,
                ),
                **kw
            )]
        
        self._do = []
        
        if start: self.start()

    def func(self, input_queue, output_queue, error_queue, error_queue_lock):
        import sys
        try:
            while not input_queue.empty():
                index, args, kwargs = input_queue.get()
                output_queue.put([index, self._target(*args, **kwargs)])
        except Exception as e:
            # Print the first exception only one time. This will show where the
            # original exception came from.
            with error_queue_lock:
                if error_queue.qsize() == 0:
                    import starsmashertools.helpers.stacktrace
                    print(starsmashertools.helpers.stacktrace.format_exc(
                        exception = e,
                        for_raise = True,
                    ))
                    print(type(e).__qualname__ + ": " + str(e))
                    error_queue.put(e)
            sys.exit(1)
    
    def get_progress(self):
        inputs = self._input_queue.qsize()
        outputs = self._output_queue.qsize()
        if inputs + outputs > 0:
            return outputs / float(inputs + outputs)
        return 0.
    
    def start(self):
        for process in self._processes:
            process.start()

    def terminate(self):
        for process in self._processes:
            process.terminate()

    def do(
            self,
            interval : float | int,
            method : typing.Callable,
            args : list | tuple = (),
            kwargs : dict = {},
    ):
        """
        A convenience function for executing the given method at regular
        intervals while waiting for :func:`~.get_output` to finish. This must be
        called prior to calling :func:`~.get_output`.

        Parameters
        ----------
        interval : float, int
            The interval to perform `method` on.

        method : callable
            The method to call.

        Other Parameters
        ----------------
        args : list, tuple, default = ()
            Positional arguments to pass to `method`.

        kwargs : dict, default = {}
            Keyword arguments to pass to `method`.
        """
        self._do += [[interval, method, args, kwargs]]

    def get_output(self):
        """
        Obtain all outputs and identifiers of the target function specified in 
        :func:`~.__init__`. Blocks until all processes have finished.
        
        Returns
        -------
        list
            A list of results from the target function specified in
            :func:`~.__init__` in the same order as the input positional and 
            keyword arguments.
        """
        import time
        import sys

        # Block until all processes have finished
        timers = [0.] * len(self._do)
        t0 = time.time()
        Ndone = 0
        while self._output_queue.qsize() < self._expected_outputs:
            if not self._error_queue.empty():
                self.terminate()
                sys.exit(1)
            
            t1 = time.time()
            deltaTime = t1 - t0
            t0 = t1
            for i, d in enumerate(self._do):
                timers[i] += deltaTime
                if timers[i] >= d[0]:
                    timers[i] -= d[0] # Maintain remainder
                    d[1](*d[2], **d[3])
            time.sleep(1.e-4)
        
        self.terminate()
        
        outputs = []
        indices = []
        for i in range(self._output_queue.qsize()):
            index, output = self._output_queue.get()
            outputs += [output]
            indices += [index]
        
        # Return sorted by input indices
        return [x for _, x in sorted(zip(indices, outputs))]
