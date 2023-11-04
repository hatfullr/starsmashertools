import starsmashertools.helpers.string as string
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
import starsmashertools.helpers.file
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.readonlydict
import copy

class Input(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
    """
    This class holds information about inputs that are sent to a StarSmasher
    simulation. It is a dictionary whose keys are StarSmasher code variable
    names and the values are the actual values that those variables have in
    StarSmasher at runtime. We accomplish this by reading the StarSmasher source
    code available in the simulation directory as well as the input list
    (usually called 'sph.input') to determine each value.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(self, directory : str):
        self.directory = directory
        self._src = None
        self._initialized = False

    @property
    def src(self):
        """
        The StarSmasher source directory.
        """
        if self._src is None:
            self._src = path.get_src(self.directory, throw_error=True)
        return self._src

    @api
    def __getitem__(self, item, **kwargs):
        if item not in list(self.keys()) and not self._initialized: self.initialize()
        return super(Input, self).__getitem__(item, **kwargs)

    def get_init_file(self):
        """
        Obtain the "init.f" file that StarSmasher uses to initialize the
        simulation.
        """
        return path.realpath(path.join(self.src, preferences.get_default(self, 'src init filename', throw_error=True)))
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_input_filename(
            self,
            init_file : str | type(None) = None,
    ):
        """
        Open up the init.f file to figure out what it reads as the input file.
        We search for the "namelist" block in the variable declarations of
        init.f and then search the simulation directory's files for one whose
        first line matches the namelist name.

        Parameters
        ----------
        init_file : str, None, default = None
            A path to the init.f file in the StarSmasher source code. If `None`,
            then the value is set to the return value of
            :func:`~.get_init_file`.

        Returns
        -------
        str
            A path to the name of the input file, which is "sph.input" by
            default unless "init.f" in the StarSmasher source code has been
            modified.
        """
        if init_file is None: init_file = self.get_init_file()

        inSubroutine = False
        namelist_name = None
        filename_listening = False
        input_filename = None
        content = []
        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            for line in f:
                ls = line.strip()

                # Skip empty lines
                if not ls: continue

                # Always skip comments
                if line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                    continue

                if ls.lower().startswith('subroutine get_input'):
                    inSubroutine = True

                # Read until we get to the namelist we are looking for
                if not inSubroutine: continue
                
                if ls.lower() in ['end', 'end subroutine']:
                    inSubroutine = False
                    break

                # ----------
                # Below this point we are inside the get_input subroutine

                if ls.lower().startswith('namelist'):
                    namelist_name = ls.lower().split('/')[1]

                # Keep reading until we locate the namelist
                if namelist_name is None: continue
                
                if not filename_listening:
                    if ls.lower().startswith('open('):
                        # Start listening for the filename we want to find
                        filename_listening = True
                        content += [ls]
                        continue
                else: # We are listening for a filename
                    content += [ls]
                    if ls.lower().startswith('close('):
                        filenum = content[0].lower().replace('open(','').split(',')[0]
                        for l in content:
                            if l.lower().startswith('read(%s' % filenum):
                                if l.lower().split(',')[1].replace(')','') == namelist_name:
                                    _f = content[0].lower().split('file')[1].split(',')[0].replace('=','').replace("'",'').replace('"','').strip()
                                    idx0 = content[0].lower().index(_f)
                                    idx1 = idx0 + len(_f)
                                    input_filename = content[0][idx0:idx1]
                                    break
                        else: continue
                        # We have located the input filename
                        break

                    
        if input_filename is None:
            raise Exception("Failed to find the input filename in '%s'" % init_file)
        return input_filename

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_default_values(self, init_file : str | type(None) = None):
        if init_file is None: init_file = self.get_init_file()

        input_filename = self.get_input_filename(init_file = init_file)
        
        obj = {}
        
        # Read the init.f file to obtain default values
        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            lines = f.read().split("\n")


        # Isolate the get_input subroutine
        for i, line in enumerate(lines):
            if line.strip() == 'subroutine get_input':
                subroutine_start = i
                break
        for i, line in enumerate(lines[subroutine_start:]):
            if line.strip() in ['end', 'end subroutine']:
                subroutine_stop = subroutine_start + i
                break
        lines = lines[subroutine_start:subroutine_stop]
        content = "\n".join(lines)

        # Find the namelist for the inputs
        for i, line in enumerate(lines):
            if 'namelist/' in line:
                namelist_start_idx = i
                break
        for i, line in enumerate(lines[namelist_start_idx+1:]):
            if not line.strip().startswith('$'):
                namelist_stop_idx = namelist_start_idx + i + 1
                break
        namelist = "\n".join(lines[namelist_start_idx:namelist_stop_idx])
        namelist = namelist.replace('$', '')
        namelist = namelist.split('/')[2]

        # Get the namelist variable names
        namelist_variables = [item.strip().lower() for item in namelist.replace('\n','').split(',')]

        # Search the remaining body of the code for variable assignments
        body = lines[namelist_stop_idx:]
        for i, line in enumerate(body):
            ls = line.strip()

            # Ignore empty lines
            if len(ls) == 0: continue

            # Ignore comments
            if line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                continue

            # Stop when StarSmasher opens the input file to overwrite defaults
            if 'open(' in ls.lower() and input_filename in ls:
                break

            if '=' in ls:
                # Skip leading '!' fortran comments
                if '!' in ls and ls.index('!') < ls.index('='): continue
                
                send = '='.join(ls.split('=')[1:])
                to_eval = starsmashertools.helpers.string.fortran_to_python(send)
                key = ls.split('=')[0].lower().strip()
                
                # Skip non-namelist members
                if key not in namelist_variables: continue
                obj[key] = eval(to_eval.replace("!","#"), {}, obj)
        return obj

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_input_values(
            self,
            defaults : dict | type(None) = {},
            filename : str | type(None) = None,
            init_file : str | type(None) = None,
    ):
        if init_file is None:
            init_file = self.get_input_filename(init_file = init_file)
        if defaults is None:
            defaults = self.get_default_values(init_file = init_file)
        if filename is None:
            filename = starsmashertools.helpers.path.join(
                self.directory,
                self.get_input_filename(init_file = init_file),
            )

        with starsmashertools.helpers.file.open(filename, 'r') as f:
            lines = f.read().split("\n")

        obj = copy.deepcopy(defaults)
        for line in lines:
            ls = line.strip()
            if '=' in ls:
                # Skip leading '!' fortran comments
                if '!' in ls and ls.index('!') < ls.index('='): continue
                to_eval = starsmashertools.helpers.string.fortran_to_python(line.replace(',','')).strip()
                key = to_eval.split('=')[0].lower()
                to_eval = '='.join(to_eval.split('=')[1:])
                obj[key] = eval(to_eval.replace("!","#"))
        
        return obj
                
    

    @api
    def initialize(self):
        """
        Read the StarSmasher input file and the "init.f" file to determine the
        value of each StarSmasher input variable. Fills this object with keys
        and values.
        """
        if self._initialized: raise Exception("Cannot initialize an Input object that is already initialized")

        init_file = self.get_init_file()
        #defaults = self.get_default_values(init_file = init_file)
        inputs = self.get_input_values(defaults=None, init_file = init_file)
        
        super(Input, self).__init__(inputs)
        self._initialized = True
