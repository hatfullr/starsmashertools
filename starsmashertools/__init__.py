# Add functions here to perform actions just before the program exits.
import starsmashertools.lib.output
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import numpy as np
import contextlib
import os
import importlib.metadata as metadata
import importlib.util as util
import starsmashertools.mpl # Runs code in mpl/__init__.py
import signal

__version__ = metadata.version('starsmashertools')

if __file__.endswith(os.path.join('starsmashertools', 'starsmashertools','__init__.py')):
    SOURCE_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
else:
    SOURCE_DIRECTORY = os.path.dirname(os.path.dirname(util.find_spec('starsmashertools').origin))

# Check if some version string is older than the current version
def _is_version_older(version, other = None):
    if other is None: other = __version__
    
    # We use semantic versioning 2.0.0
    major, minor, patch = version.split('.')
    major = int(major)
    minor = int(minor)
    patch = int(patch)

    _major, _minor, _patch = other.split('.')
    _major = int(_major)
    _minor = int(_minor)
    _patch = int(_patch)

    if major < _major: return True
    if minor < _minor: return True
    if patch < _patch: return True
    return False


def _get_git_branch():
    # TODO: Test if this still works on main branch.
    import os

    filename = os.path.join(
        SOURCE_DIRECTORY,
        '.git',
        'HEAD',
    )
    with open(filename, 'r') as f:
        content = f.readline()
    return content.split('/')[-1]

# If our current version mis-matches the installed version, show a warning
def _check_version():
    # Don't do this on my local development machine. I increment the version
    # number on each commit and I don't want to run the install script every
    # time I commit b/c it's kind of slow.
    import os

    if os.path.exists('/home/hatfull/starsmashertools'):
        if os.path.samefile(SOURCE_DIRECTORY, '/home/hatfull/starsmashertools'):
            return

    import urllib.request
    import urllib.error
    import starsmashertools.helpers.warnings

    def get_latest_version():
        url = 'https://raw.githubusercontent.com/hatfullr/starsmashertools/{branch:s}/pyproject.toml'.format(
            branch = _get_git_branch(),
        )
        request = urllib.request.Request(url)
        # This is to get the most recent webpage
        request.add_header('Cache-Control', 'max-age=0')
        response = urllib.request.urlopen(request)
        contents = response.read().decode('utf-8')
        for line in contents.split('\n'):
            if line.startswith('version'):
                return line.split('=')[1].replace("'",'').replace('"','').strip()
        return None

    connected = True
    try:
        latest_version = get_latest_version()
    except Exception as e:
        # Simply don't bother checking the latest version if anything goes wrong.
        # This way the user can always still use their installed version.
        if not isinstance(e, urllib.error.URLError):
            starsmashertools.helpers.warnings.warn("Something went wrong while checking the latest version of starsmashertools while connected to the internet. This should never happen, but if it does then you can safely ignore this warning. You can filter this warning out by importing the 'warnings' package before importing starsmashertools and writing \"warnings.filterwarnings(action='ignore')\" before the starsmashertools import and \"warnings.resetwarnings()\" after the import.")
        connected = False
    if connected:
        if latest_version is None:
            starsmashertools.helpers.warnings.warn("Failed to find the version number in the pyproject.toml file downloaded from GitHub. It is uncertain if your current starsmashertools version is the most recent version.")
        elif latest_version != __version__:
            starsmashertools.helpers.warnings.warn("Current starsmashertools version is '%s', while latest version on GitHub is '%s'. If you just downloaded the latest version make sure you also run the install script." % (__version__, latest_version))

_check_version()

        
@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_simulation(directory : str):
    """
    Obtain a StarSmasher simulation from a directory.
    
    Parameters
    ----------
    directory : str
        A path to a StarSmasher simulation directory.

    Returns
    -------
    :class:`~.lib.relaxation.Relaxation`, :class:`~.lib.binary.Binary`, 
    :class:`~.lib.dynamical.Dynamical`, or :class:`~.lib.simulation.Simulation`
        The StarSmasher simulation at the given `directory`.
    
    Examples
    --------
    This example obtains a :class:`~.lib.simulation.Simulation` object from a
    directory located at "/home/me/mysimulation"::
        
        import starsmashertools
        simulation = starsmashertools.get_simulation('/home/me/mysimulation')
    
    """
    return get_type(directory)(directory)

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_type(directory : str):
    """
    Find the type of simulation located at the given directory.

    Parameters
    ----------
    directory : str
        A path to a StarSmasher simulation directory.

    Returns
    -------
    :class:`~.lib.relaxation.Relaxation`, :class:`~.lib.binary.Binary`, 
    :class:`~.lib.dynamical.Dynamical`, or :class:`~.lib.simulation.Simulation`
    """
    import starsmashertools.lib.simulation
    
    simulation = starsmashertools.lib.simulation.Simulation(directory)
    
    nrelax = simulation['nrelax']
    if nrelax == 0:
        import starsmashertools.lib.dynamical
        return starsmashertools.lib.dynamical.Dynamical
    if nrelax == 1:
        import starsmashertools.lib.relaxation
        return starsmashertools.lib.relaxation.Relaxation
    if nrelax == 2:
        import starsmashertools.lib.binary
        return starsmashertools.lib.binary.Binary
    # If it's some other type
    return starsmashertools.lib.simulation.Simulation

@api
def relaxation(*args, **kwargs):
    """
    Get a :class:`~.lib.relaxation.Relaxation` type StarSmasher simulation.

    Other Parameters
    ----------------
    *args
        Positional arguments are passed directly to the constructor method of
        :class:`~.lib.relaxation.Relaxation`.

    **kwargs
        Keyword arguments are passed directly to the constructor method of 
        :class:`~.lib.relaxation.Relaxation`.

    Returns
    -------
    :class:`~.lib.relaxation.Relaxation`
    """
    import starsmashertools.lib.relaxation
    return starsmashertools.lib.relaxation.Relaxation(*args, **kwargs)

@api
def binary(*args, **kwargs):
    """
    Get a :class:`~.lib.binary.Binary` type StarSmasher simulation.

    Other Parameters
    ----------------
    *args
        Positional arguments are passed directly to the constructor method of
        :class:`~.lib.binary.Binary`.

    **kwargs
        Keyword arguments are passed directly to the constructor method of
        :class:`~.lib.binary.Binary`.

    Returns
    -------
    :class:`~.lib.binary.Binary`
    """
    import starsmashertools.lib.binary
    return starsmashertools.lib.binary.Binary(*args, **kwargs)

@api
def dynamical(*args, **kwargs):
    """
    Get a :class:`~.lib.dynamical.Dynamical` type StarSmasher simulation.

    Other Parameters
    ----------------
    *args
        Positional arguments are passed directly to the constructor method of
        :class:`~.lib.dynamical.Dynamical`.

    **kwargs
        Keyword arguments are passed directly to the constructor method of 
        :class:`~.lib.dynamical.Dynamical`.

    Returns
    -------
    :class:`~.lib.dynamical.Dynamical`
    """
    import starsmashertools.lib.dynamical
    return starsmashertools.lib.dynamical.Dynamical(*args, **kwargs)

@api
def iterator(*args, **kwargs):
    """
    Obtain a :class:`~.lib.output.OutputIterator` that can be used to iterate
    through output files. Using an iterator in this way is usually faster than
    looping through the files normally because an
    :class:`~.lib.output.OutputIterator` asynchronously reads ahead.
    
    Other Parameters
    ----------------
    *args
        Positional arguments are passed directly to the constructor method of 
        :class:`~.lib.output.OutputIterator`

    **kwargs
        Keyword arguments are passed directly to the constructor method of 
        :class:`~.lib.output.OutputIterator`

    Returns
    -------
    :class:`~.lib.output.OutputIterator`

    See Also
    --------
    Constructor method of :class:`~.lib.output.OutputIterator`
    """
    import starsmashertools.lib.output
    return starsmashertools.lib.output.OutputIterator(*args, **kwargs)

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_particles(
        output : starsmashertools.lib.output.Output,
        particles : np.ndarray | list | tuple,
):
    """
    Mask the given output file such that it contains only the particles
    specified by `particles`.

    Parameters
    ----------
    output : :class:`~.lib.output.Output`
        The simulation output files from which to extract the particle
        information.

    particles : np.ndarray, list, tuple
        A collection of particle IDs as zeroth-indexed integers or as a numpy
        boolean array.
    
    Returns
    -------
    :class:`~.lib.output.Output`
        A copy of `output`, masked such that it includes only the specified
        particles.

    See Also
    --------
    :func:`mask`
    """
    import copy
    with mask(output, particles) as masked_output:
        return copy.deepcopy(masked_output)

@contextlib.contextmanager
@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def trace_particles(
        particles : int | list | tuple | np.ndarray,
        outputs_or_simulation : list | tuple | starsmashertools.lib.output.OutputIterator | starsmashertools.lib.simulation.Simulation,
        **kwargs
):
    import starsmashertools.lib.output
    
    if isinstance(outputs_or_simulation, (list, tuple)):
        simulation = outputs_or_simulation[0].simulation
        outputs = starsmashertools.lib.output.ParticleIterator(
            particles,
            [o.path for o in outputs_or_simulation],
            simulation,
            **kwargs
        )
    elif isinstance(outputs_or_simulation, starsmashertools.lib.output.OutputIterator):
        outputs = starsmashertools.lib.output.ParticleIterator(
            particles,
            outputs_or_simulation.filenames,
            outputs_or_simulation.simulation,
            onFlush = outputs_or_simulation.onFlush,
            max_buffer_size = outputs_or_simulation.max_buffer_size,
            asynchronous = outputs_or_simulation.asynchronous,
            **outputs_or_simulation.kwargs
        )
        simulation = outputs_or_simulation.simulation
    else:
        simulation = outputs_or_simulation
        outputs = simulation.get_output_iterator(**kwargs)
        outputs = starsmashertools.lib.output.ParticleIterator(
            particles,
            outputs.filenames,
            outputs.simulation,
            onFlush=outputs.onFlush,
            max_buffer_size=outputs.max_buffer_size,
            asynchronous = outputs.asynchronous,
            **outputs.kwargs
        )

    try:
        yield outputs
    finally:
        pass

    

    
@contextlib.contextmanager
@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def mask(
        output : starsmashertools.lib.output.Output,
        mask : np.ndarray | list | tuple,
):
    """
    Apply a mask to the given :class:`~.lib.output.Output` object that hides
    selected particles.

    Parameters
    ----------
    output : :class:`~.lib.output.Output`
        The StarSmasher output file to mask.

    mask : np.ndarray, list, tuple
        The mask to apply to `output`, which can be either a collection of 
        zeroth-indexed integers representing the particle IDs, or a collection
        of ``bool`` values the same length as the number of particles in
        `output`.

    Returns
    -------
    :class:`~.lib.output.Output`
        The masked output file.

    Examples
    --------
    This example prints the x positions of particles 0 and 4000 from the first 
    output file in the current working directory::

        import starsmashertools
        simulation = starsmashertools.get_simulation('.')
        output = starsmashertools.get_output(0)
        with starsmashertools.mask(output, [0, 4000]) as masked:
            print(masked['x'])

    
    """
    if output._mask is not None:
        raise Exception("Cannot mask output that is already masked: '%s'" % str(output.path))

    # Mask the data
    output.mask(mask)
    try:
        yield output
    finally:
        # Unmask the data
        output.unmask()

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def interpolate(
        outputs : "list | tuple | starsmashertools.lib.output.OutputIterator",
        values : list,
):
    """
    Returns a function that performs a 1D linear interpolation between the
    provided output files on the time axis for a specific value.

    Parameters
    ----------
    outputs : list, tuple, :class:`~.lib.output.OutputIterator`
        The outputs to interpolate.

    values : list, tuple
        The Output keys to interpolate.

    Returns
    -------
    function
        A callable function that accepts a time as input and returns a
        dictionary whose keys are that of the `values` arguments and values are
        particle quantities interpolated at the given time. Raises a
        ``ValueError`` if the input time is out of bounds of the interpolation.
    """
    import starsmashertools.math
    import numpy as np
    
    if len(outputs) < 2:
        raise ValueError("You must provide 2 or more outputs for interpolation")

    if isinstance(outputs, starsmashertools.lib.output.OutputIterator):
        simulation = outputs.simulation
        outputs.kwargs['return_headers'] = True
        outputs.kwargs['return_data'] = True
        outputs.kwargs['asynchronous'] = True
    else:
        simulation = outputs[0].simulation
        filenames = []
        for o in outputs:
            if o.simulation != simulation:
                raise ValueError("All Output objects must be a member of the same Simulation. Found '%s' is a member of simulation '%s', but expected simulation '%s'" % (str(o), o.simulation.directory, simulation.directory))
            filenames += [o.path]
        outputs = starsmashertools.lib.output.OutputIterator(
            filenames,
            simulation,
            return_headers = True,
            return_data = True,
            asynchronous = True,
        )
            
    # We should be able to comfortably store all the output headers
    times = []
    vals = []
    for output in outputs:
        times += [output.header['t']]
        vals += [{v:output[v] for v in values}]
    times = np.asarray(times)
    tmin = np.amin(times)
    tmax = np.amax(times)

    
    def interp(time, *which):
        if isinstance(time, starsmashertools.lib.units.Unit):
            # Convert the unit to simulation units
            time = time.convert(simulation.units.time.label)

        if time > tmax or time < tmin:
            raise ValueError("Time %s is out of range. Must be between %f and %f" % (str(time), tmin, tmax))

        left = np.argmin(np.abs(times - time))
        if left + 1 >= len(times): left -= 1
        right = left + 1

        left = vals[left]
        right = vals[right]

        if which: keys = which
        else: keys = values

        x = [times[left], times[right]]

        results = {}
        for key in keys:
            y = [vals[left][key], vals[right][key]]
            results[key] = starsmashertools.math.linear_interpolate(
                x,
                y,
                time,
            )
        return results
    return interp
























def _get_decorators():
    import os
    import ast
    
    tocheck = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    files = []
    for dirpath, dirnames, filenames in os.walk(tocheck):
        for filename in filenames:
            if not filename.endswith(".py"): continue
            path = os.path.join(dirpath, filename)
            files += [path]
            

    def get_full_function_name(_file,  name, _class = None):
        path = get_module_name(_file, name)
        name = get_function_name(name, _class = _class)
        return '%s.%s' % (path, name)

    def get_module_name(_file, name):
        path = os.path.relpath(_file, SOURCE_DIRECTORY)
        return path.replace(".py",'').replace(os.sep, '.')

    def get_function_name(name, _class = None):
        path = get_module_name(_file, name)
        if _class is None: return name
        return '%s.%s' % (_class, name)

    def get_modules(_file):
        class ImportNodeVisitor(ast.NodeVisitor):
            def __init__(self, *args, **kwargs):
                super(ImportNodeVisitor, self).__init__(*args, **kwargs)
                self.modules = {}
                self.module_names = {}
            
            def visit_Import(self, node):
                for name in node.names:
                    asname = name.asname
                    if not asname: asname = name.name
                    self.modules[name.name] = asname
                    self.module_names[name.name] = name.name
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                for name in node.names:
                    asname = name.asname
                    if not asname: asname = name.name
                    full_name = '%s.%s' % (node.module, name.name)
                    self.modules[full_name] = asname
                    self.module_names[full_name] = node.module
                self.generic_visit(node)
        with open(_file, 'r') as f:
            tree = ast.parse(f.read(), _file)
        nodeVisitor = ImportNodeVisitor()
        nodeVisitor.visit(tree)
        return nodeVisitor.modules, nodeVisitor.module_names

    functions = {}
    for _file in files:
        basename = os.path.basename(_file)
        if (basename.endswith("#") or basename.endswith("~") or
            basename.startswith("#") or basename.startswith(".#")): continue
        
        
        with open(_file, 'r') as f:
            modules, module_names = get_modules(_file)
            
            listening_for_def = False
            detected_decorators = []
            _class = None
            _class_indent = None
            in_comment_block = False
            for line in f:
                # Remove comments
                ls = line.strip()
                if not ls: continue
                if '"""' in ls:
                    in_comment_block = not in_comment_block
                    continue
                if in_comment_block: continue
                if '#' in ls:
                    ls = ls[:ls.index('#')]
                    if not ls: continue
                
                indent = line.index(ls[0])

                if _class is not None and indent <= _class_indent:
                    # If the indentation has moved out of the class then we need
                    # to fall back down 1 class level
                    if _class.count('.') == 0:
                        # No more nested classes
                        _class = None
                        _class_indent = None
                    else:
                        # Still in a class statement, dropping out of one level
                        # of nested classes
                        _class = '.'.join(_class.split('.')[:-1])
                        _class_indent = indent
            
                if ls.startswith('class'):
                    l = ls.replace('class', '').strip()
                    if '(' in l: l = l[:l.index("(")]
                    if ':' in l: l = l[:l.index(':')]

                    # When we find a class, we need to add the '.stuff' syntax
                    # to _class only if the current indentation level is greater
                    # than the previous.
                    # If we already have a '.' syntax, then we need to remove
                    # the final '.' member and replace it with the class we just
                    # found.

                    if _class is None:
                        _class = l
                        _class_indent = indent
                    else:
                        if '.' in _class:
                            if indent > _class_indent:
                                _class = '.'.join([_class,l])
                            else:
                                _class = '.'.join(_class.split('.')[:-1] + [l])
                            _class_indent = indent
                    continue
                
                if listening_for_def:
                    if ls.startswith('def'):
                        l = ls.replace('def', '').strip()
                        name = l[:l.index("(")]
                        full_name = get_full_function_name(_file, name, _class=_class)
                        for detected_decorator in detected_decorators:

                            module = get_module_name(_file, name)
                            if modules:
                                for key, val in modules.items():
                                    if detected_decorator == val:
                                        detected_decorator = key
                                        module = module_names[key]
                                        break
                            
                            if detected_decorator not in functions.keys():
                                functions[detected_decorator] = []
                            functions[detected_decorator] += [{
                                'full name' : full_name,
                                'short name' : name,
                                'module' : module,
                                'class' : _class,
                            }]
                        detected_decorators = []
                        listening_for_def = False
                    elif ls.startswith("@"):
                        l = ls.replace("@", '').strip()
                        if '(' in l: l = l[:l.index("(")]
                        if modules:
                            for key, val in modules.items():
                                if l == val:
                                    l = key
                                    break
                        detected_decorators += [l]
                    else:
                        raise NotImplementedError("Detected decorators that are floating above a function definition or don't belong to any function definition in '%s'" % _file)
                else:
                    if ls.startswith("@"):
                        listening_for_def = True
                        l = ls.replace("@",'').strip()
                        if '(' in l: l = l[:l.index("(")]
                        if modules:
                            for key, val in modules.items():
                                if l == val:
                                    l = key
                                    break
                        detected_decorators += [l]
    return functions












































# Cleanup
try: del api
except: pass

try: del np
except: pass

try: del contextlib
except: pass

try: del metadata
except: pass

try: del util
except: pass

try: del starsmashertools
except: pass

try: del os # The most terrifying syntax...
except: pass

try: del f
except: pass
