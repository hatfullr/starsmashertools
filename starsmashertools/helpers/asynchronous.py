import multiprocessing
import multiprocessing.queues
import typing
import threading

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
    supply a target function which will be run on many individual processes.
    """
    
    def __init__(
            self,
            target : typing.Callable,
            args : tuple | list | type(None) = None,
            kwargs : tuple | list | type(None) = None,
            nprocs : int | type(None) = None,
            start : bool = True,
            progress_message : "starsmashertools.helpers.string.ProgressMessage | type(None)" = None,
            **kw
    ):
        """
        Initializer. If there are no available processes (because too many have
        already been spawned), whenever processes become available we create 
        processes up to a number ``nprocs``\, which must be smaller than
        ``multiprocessing.cpu_count()``\.
        
        Parameters
        ----------
        target : callable
            The function to be executed in parallel.

        args : list, tuple, None, default = None
            A list of positional arguments to pass to ``target``\. Each element
            should be a list of positional arguments. The number of calls to
            ``target`` is equal to the number of elements in this list. If
            `None`\, no arguments will be passed to ``target``\.

        kwargs : list, tuple, None, default = None
            A list of keyword arguments to pass to ``target``\. Each element
            should be a dictionary of keyword arguments. If ``args`` is not 
            `None` and this list has fewer elements than ``args``\, then ``{}`` 
            will be appended until it is the same length as ``args``\. 
        

        Other Parameters
        ----------------
        nprocs : int, None, default = None
            The number of processes to use. If `None`\, the number of processes
            will be the number of available CPU cores as given by
            ``multiprocessing.cpu_count()``\.

        start : bool, default = True
            Automatically start the processes after initialization without
            blocking. To start with blocking, specify `False` and then call
            :meth:`~.start` in your code.

        progress_message : :class:`~.helpers.string.ProgressMessage`\, None, default = None
            A progress message to update each time a function call is completed
            by one of the child processes.

        **kw
            Other keyword arguments are passed directly to
            ``multiprocessing.Process`` when creating each individual process.
            By default, keyword 'daemon' is set to `True`\.
        """
        if nprocs is None: nprocs = max_processes
        self._nprocs = nprocs

        self.progress_message = progress_message
        
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

        # Don't spawn more processes than we need to
        nprocs = min(nprocs, len(args))
        
        manager = multiprocessing.Manager()
        self._input_queue = manager.Queue()
        self._output_queue = manager.Queue()
        self._error_queue = manager.Queue()
        
        self._error_queue_lock = manager.Lock()

        self._expected_outputs = 0
        for i, (a, k) in enumerate(zip(args, kwargs)):
            self._input_queue.put([i, a, k])
            self._expected_outputs += 1
        
        self._target = target
        args = [
            self._input_queue,
            self._output_queue,
            self._error_queue,
            self._error_queue_lock,
        ]
        self._processes = []
        self._create_processes(nprocs, self.func, args, **kw)
        
        self._do = []
        
        if start: self.start()

    def _create_processes(self, nprocs, target, args, **kw):
        if self._processes:
            raise Exception("Cannot call _create_processes more than one time")
        
        for i in range(nprocs):
            self._processes += [Process(
                target = target,
                args = args,
                **kw
            )]

    def func(self, input_queue, output_queue, error_queue, error_queue_lock):
        import sys

        try:
            while not input_queue.empty():
                index, args, kwargs = input_queue.get()
                result = self._target(*args, **kwargs)
                output_queue.put([index, result])
        except Exception as e:
            # Print the first exception only one time. This will show where
            # the original exception came from.
            try:
                if isinstance(e, (BrokenPipeError, ConnectionResetError)): raise
                with error_queue_lock:
                    if error_queue.qsize() == 0:
                        import starsmashertools.helpers.stacktrace
                        print(starsmashertools.helpers.stacktrace.format_exc(
                            exception = e,
                            for_raise = True,
                        ))
                        print(type(e).__qualname__ + ": " + str(e))
                        error_queue.put([e, index, args, kwargs])
            except (BrokenPipeError, ConnectionResetError):
                import starsmashertools.helpers.stacktrace
                print(starsmashertools.helpers.stacktrace.format_exc(
                    exception = e,
                    for_raise = True,
                ))
                print(type(e).__qualname__ + ": " + str(e))
                sys.stderr.close()
            sys.exit(1)
    
    def get_progress(self):
        if self._expected_outputs > 0:
            return (self._expected_outputs - self._input_queue.qsize()) / float(self._expected_outputs)
        return 0.
    
    def start(self):
        try:
            for process in self._processes:
                process.start()
        except (BrokenPipeError, ConnectionResetError):
            quit()

    def terminate(self):
        # If we terminate without joining, the queue becomes corrupted and
        # unusable, possibly causing memory leaks.
        #for process in self._processes:
        #    process.join()
            
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
        intervals while waiting for :meth:`~.get_output` to finish. This must be
        called prior to calling :meth:`~.get_output`\.

        Parameters
        ----------
        interval : float, int
            The interval to perform `method` on.

        method : callable
            The method to call.

        Other Parameters
        ----------------
        args : list, tuple, default = ()
            Positional arguments to pass to `method`\.

        kwargs : dict, default = {}
            Keyword arguments to pass to `method`\.
        """
        self._do += [[interval, method, args, kwargs]]

    def process_error(self, from_queue):
        import contextlib, sys
        error, index, args, kwargs = from_queue
        self.terminate()
        print("ParallelFunction encountered an error at index %d.\nArguments = %s\nKeywords = %s" % (index, str(args), str(kwargs)))
        with contextlib.suppress(Exception):
            raise(error)
        sys.exit(1)

    def get_output(self, sort : bool = True):
        """
        Obtain all outputs and identifiers of the target function specified in 
        :meth:`~.__init__`\. Blocks until all processes have finished. If you
        want the results to be sorted then this will return a list of results.
        Otherwise, this returns an iterator which blocks on every iteration
        until a result is ready to be given. Thus, when `sort` is False you
        should loop over the iterator even if you don't plan to use the returned
        values.

        Parameters
        ----------
        sort : bool, default = True
            If `False`\, return an iterator that is not sorted. Otherwise, obtain
            all results and put them in a list which is sorted by the same order
            as the arguments and keywords given in
            :meth:`~.ParallelFunction.__init__`\.
        
        Returns
        -------
        iterator or list
            If `sort` is False, returns an iterator which blocks on every 
            iteration until a value from the processes is ready to be returned.
            Otherwise, returns a list of all the returned values only after all
            processes have finished, sorted by their indices (same order as the
            input positional and keyword arguments).
        """
        import time
        
        if not sort:
            return ParallelFunction.ResultIterator(
                self._output_queue,
                self._expected_outputs,
                error_queue = self._error_queue,
                onError = self.process_error,
                onFinished = self.terminate,
                do_every = self._do,
                progress_message = self.progress_message,
            )
        else:
            # Block until all processes have finished
            timers = [0.] * len(self._do)
            t0 = time.time()
            previous_qsize = self._output_queue.qsize()
            while self._output_queue.qsize() < self._expected_outputs:
                if not self._error_queue.empty():
                    self.process_error(self._error_queue.get())

                t1 = time.time()
                deltaTime = t1 - t0
                t0 = t1
                for i, d in enumerate(self._do):
                    timers[i] += deltaTime
                    if timers[i] >= d[0]:
                        timers[i] -= d[0] # Maintain remainder
                        d[1](*d[2], **d[3])

                if self.progress_message:
                    current_qsize = self._output_queue.qsize()
                    if current_qsize != previous_qsize:
                        self.progress_message.increment(
                            amount = current_qsize - previous_qsize,
                        )
                        previous_qsize = current_qsize
                        
                time.sleep(1.e-4)

            self.terminate()

            outputs = []
            indices = []
            for i in range(self._output_queue.qsize()):
                index, output = self._output_queue.get()
                outputs += [output]
                indices += [index]
            
            # Close the queues
            self.close_queues()
            
            # Return sorted by input indices
            return [x for _, x in sorted(zip(indices, outputs))]

    def close_queues(self):
        queues = [self._input_queue,self._output_queue,self._error_queue]
        for queue in queues:
            while not queue.empty(): queue.get()

    class ResultIterator(object):
        def __init__(
                self,
                queue : multiprocessing.queues.Queue,
                expected_length : int,
                error_queue : multiprocessing.queues.Queue | type(None) = None,
                onError : typing.Callable | type(None) = None,
                onFinished : typing.Callable | type(None) = None,
                do_every : list | tuple = [],
                progress_message : "starsmashertools.helpers.string.ProgressMessage | type(None)" = None,
        ):
            self.queue = queue
            self.expected_length = expected_length
            self.error_queue = error_queue
            self.onError = onError
            self.do_every = do_every
            self.onFinished = onFinished
            self.progress_message = progress_message
            
            self._iteration = 0
            self._timers = [0.] * len(self.do_every)
            self._t0 = None
            
        def __iter__(self): return self

        def __next__(self):
            import time

            if self._iteration >= self.expected_length:
                self.stop()

            if self._iteration == 0:
                self._t0 = time.time()
            
            # Block until we have a member in the queue to get. Check the error
            # queue for any errors that need raised.
            while self.queue.qsize() == 0:
                if (self.error_queue is not None and
                    not self.error_queue.empty()):
                    if self.onError is not None:
                        self.onError(self.error_queue.get())
                        self.stop()

                t1 = time.time()
                deltaTime = t1 - self._t0
                self._t0 = t1
                for i, d in enumerate(self.do_every):
                    self._timers[i] += deltaTime
                    if self._timers[i] >= d[0]:
                        self._timers[i] -= d[0] # Maintain remainder
                        d[1](*d[2], **d[3])
            
            self._iteration += 1
            ret = self.queue.get()
            if self.progress_message: self.progress_message.increment()
            return ret

        def stop(self):
            if self.onFinished is not None:
                self.onFinished()
            raise StopIteration



class ParallelIterator(ParallelFunction, object):
    """
    Perhaps a more simple concept than in :class:`~.ParallelFunction`\. This
    class spawns processes and then works as an iterator. You feed it with a
    function and the arguments and keywords you want it to process. When you
    start iteration, the processes start running, moving along the queue in
    ascending order. An iteration step is taken whenever the corresponding 
    results are ready.
    """

    def __init__(
            self,
            *args,
            buffer_size : int = 10,
            **kwargs
    ):
        self.buffer_size = buffer_size
        self._index = 0
        self._started = False
        self._buffer = []
        kwargs['start'] = False
        super(ParallelIterator, self).__init__(*args, **kwargs)

    def get_output(self, *args, **kwargs):
        raise NotImplementedError

    def _create_processes(self, nprocs, target, args, **kw):
        if self._processes:
            raise Exception("Cannot call _create_processes more than one time")
        
        self._processes = []
        for i in range(nprocs):
            self._processes += [Process(
                target = target,
                args = args + [self.buffer_size],
                **kw
            )]
    
    def func(self, input_queue, output_queue, error_queue, error_queue_lock, buffer_size):
        import time
        # Wait for the buffer to stop being full
        while output_queue.qsize() >= buffer_size:
            time.sleep(1.e-6)
        
        return super(ParallelIterator, self).func(
            input_queue, output_queue, error_queue, error_queue_lock
        )

    def start(self, *args, **kwargs):
        self._started = True
        return super(ParallelIterator, self).start(*args, **kwargs)
    
    def __iter__(self): return self
    
    def __next__(self):
        import time
        
        if self._index == 0 and not self._started:
            self.start()

        if self._index >= self._expected_outputs:
            self.terminate()
            self.close_queues()
            raise StopIteration()
        
        toremove = None
        while toremove is None:
            time.sleep(1.e-4)
            
            if not self._output_queue.empty():
                # Catch the output in the buffer
                self._buffer += [self._output_queue.get()]

            if self._buffer:
                # Check the buffer for the index we are seeking
                for i, result in enumerate(self._buffer):
                    index, output = result
                    if index != self._index: continue
                    # Found the right index
                    toremove = result
                    break
        
        index, output = toremove
        self._buffer.remove(toremove) # Remove it from the buffer
        self._index += 1 # Move to the next index
        return output
        
        
            
class Ticker(threading.Thread):
    """
    A helpful class that performs some action on time intervals.
    """
    def __init__(
            self,
            interval : int | float,
            target : typing.Callable | type(None) = None,
            args : list | tuple = (),
            kwargs : dict = {},
            delay : int | float = 0,
            limit : int | float | type(None) = None,
            after_delay : typing.Callable | type(None) = None,
            after_delay_args : list | tuple = (),
            after_delay_kwargs : dict = {},
    ):
        self.interval = interval
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.delay = delay
        self.limit = limit
        self.after_delay = after_delay
        self.after_delay_args = after_delay_args
        self.after_delay_kwargs = after_delay_kwargs
        self._stopEvent = threading.Event()
        super(Ticker, self).__init__()
        self.daemon = True
        self._cycle = None
        self.ran = False
        self.completed = False
        self.paused = False
    
    def cancel(self):
        self._stopEvent.set()
        self.join(timeout = 0)
        self.completed = False

    def cycle(
            self,
            length : int,
            args : list | tuple | type(None) = None,
            kwargs : list | tuple | dict = {},
    ):
        if self.target is None:
            raise Exception("Cannot create a cycle when 'target' is None")
        
        if args is not None:
            if len(args) > 1 and all([isinstance(a, (list, tuple)) for a in args]):
                if len(args) != length:
                    raise ValueError("Argument 'args' must have the same length as 'length' when multiple argument lists are given")
            else:
                args = [args]*length
        if isinstance(kwargs, dict): kwargs = [kwargs]*length
        self._cycle = [[self.target, a, k] for a,k in zip(args, kwargs)]

    def pause(self): self.paused = True
    def resume(self): self.paused = False

    def run(self):
        import time
        self.ran = False
        self.completed = False
        while self.paused: pass
        self._stopEvent.wait(max(self.delay - self.interval, 0))
        while self.paused: pass

        timer = 0.
        t0 = time.time()
        iteration = 0
        first = True
        while not self._stopEvent.wait(self.interval):
            while self.paused: pass
            if first:
                if self.after_delay is not None:
                    self.after_delay(*self.after_delay_args, **self.after_delay_kwargs)
                first = False
                self.ran = True
            if self.limit is not None:
                timer += time.time() - t0
                t0 = time.time()
                if timer >= self.limit:
                    self.completed = True
                    break
            
            if self.target is None: continue
            if self._cycle:
                target, args, kwargs = self._cycle[iteration]
                target(*args, **kwargs)
                iteration += 1
                if iteration >= len(self._cycle): iteration = 0
            else:
                self.target(*self.args, **self.kwargs)
