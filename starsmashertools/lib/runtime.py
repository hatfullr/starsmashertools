import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import starsmashertools.lib.simulation
import numpy as np


class Runtime(object):
    r"""
    Contains information about the processes and meta information for a
    :class:`~.lib.simulation.Simulation`\, regarding specifically things that
    occurred during when the simulation was running, such as, e.g., total wall
    time.
    """

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            simulation : starsmashertools.lib.simulation.Simulation
    ):
        self.simulation = simulation

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_wall_durations(
            self,
            include_joined : bool = False,
    ):
        r"""
        Get the amount of physical time that the simulation spent running, for
        each log file it produced.

        Other Parameters
        ----------------
        include_joined : bool, default = False
            If ``True``\, the result includes the wall times of all joined
            simulations. Otherwise, the result is only the wall time of this
            simulation directory.

        Yields
        ------
        float, :class:`~.lib.logfile.LogFile`
            The physical duration and associated log file.
        """
        import starsmashertools.lib.units
        import starsmashertools.helpers.path
        import starsmashertools.math
        
        logfiles = self.simulation.get_logfiles(include_joined = include_joined)
        if not logfiles:
            raise FileNotFoundError("No log files found in simulation '%s'" % self.simulation.directory)
        
        # We should sort the logfiles by their iteration numbers
        first_iterations = []
        for logfile in logfiles:
            first = logfile.get_first_iteration()
            if first is None: first_iterations += [float('NaN')]
            else: first_iterations += [first['iteration']]
        first_iterations = np.asarray(first_iterations)
        
        logfiles = [logfiles[i] for i in np.argsort(first_iterations) if np.isfinite(first_iterations[i])]
        
        outputs = self.simulation.get_output(include_joined = include_joined)
        basenames = [starsmashertools.helpers.path.basename(
            output.path
        ) for output in outputs]
        
        # Simulations can be stopped and restarted. Thus, we cannot directly
        # rely on the file modification times. Instead, we must check each log
        # file.
        durations = np.zeros(len(logfiles))
        for i, logfile in enumerate(logfiles):
            first = logfile.get_first_output_file(throw_error = False)
            if first is None:
                # The code may have ran but not produced any output files. In
                # this case, check to see if a restartrad.sph file was produced
                # during this log file.

                with logfile.get_buffer() as buffer:
                    if buffer.find(
                            'checkpt: writing local checkpt file at nit='.encode('utf-8'),
                            len(logfile.header),
                    ) == -1: # Failed to find phrase
                        # This log file contributes nothing to the wall time.
                        continue

                # Here the log file is for a run that produced no output, but
                # did advance the simulation. We have no way of knowing how long
                # this part of the run took.
                yield np.nan, logfile
                continue

            # "first" is only the output file's basename. It should be unique,
            # though...
            try:
                index = basenames.index(first)
            except ValueError as e:
                raise FileNotFoundError("Log file '%s' claims to have created output '%s', but that file does not exist in the simulation directory '%s'" % (logfile.path, first, self.simulation.directory)) from e

            start_timestamp = starsmashertools.helpers.path.getmtimens(
                outputs[index].path,
            ) * 1.e-9
            end_timestamp = starsmashertools.helpers.path.getmtimens(
                logfile.path,
            ) * 1.e-9

            yield end_timestamp - start_timestamp, logfile

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_wall_time(
            self,
            include_joined : bool = False,
    ):
        r"""
        Get the total amount of physical time that the simulation spent running.
        
        Other Parameters
        ----------------
        include_joined : bool, default = False
            If ``True``\, the result includes the wall times of all joined
            simulations. Otherwise, the result is only the wall time of this
            simulation directory.
        
        Returns
        -------
        walltime : :class:`~.units.Unit`
            The total wall time.
        
        Raises
        ------
        :py:class:`IndexError`
            This is raised when the simulation is missing output files.
        
        :py:class:`FileNotFoundError`
            This is raised when the simulation is missing log files. This is
            also raised if the first output file in a log cannot be located in
            the simulation directory.
        
        Notes
        -----
        File modification times in a simulation are expected to be preserved. If
        the modification times are not preserved, then this method will return
        erroneous results.

        Each time a simulation starts, it produces its first output file on the
        first iteration always. This method relies on that fact.

        Unfortunately, there is no way to capture the initialization (init.f)
        time, as StarSmasher does not create any trackable files at the start of
        the initialization. As such, the results of this method do not include
        the code initialization times.

        If a log file produced no output files, then the simulation may still
        have produced a restartrad.sph checkpoint file during the run. Such a 
        log is checked for the phrase 'checkpt: writing local checkpt file at 
        nit='. If that phrase cannot be found, it is assumed that the simulation
        did not advance at all while writing that log file, and so contributes
        nothing to the wall time.
        """
        import starsmashertools.math
        import starsmashertools.lib.units

        durations = []
        for duration, logfile in self.get_wall_durations(include_joined=False):
            durations += [duration]
        durations = np.asarray(durations)
        
        # linear interpolate the durations for which a log file wrote a
        # checkpoint file, but did not write any output files.
        durations[~np.isfinite(durations)] = starsmashertools.math.linear_interpolate(
            np.arange(len(durations))[np.isfinite(durations)],
            durations[np.isfinite(durations)],
            np.arange(len(durations))[~np.isfinite(durations)],
        )
        
        walltime = starsmashertools.lib.units.Unit(sum(durations), 's')
        
        if include_joined:
            for simulation in self.simulation.joined_simulations:
                walltime += simulation.runtime.get_wall_time(
                    include_joined = False,
                )
        return walltime
