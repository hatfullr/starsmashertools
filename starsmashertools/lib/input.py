import starsmashertools.helpers.string
import starsmashertools.helpers.path
import starsmashertools.preferences
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
            self._src = starsmashertools.helpers.path.get_src(self.directory, throw_error=True)
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
        return starsmashertools.helpers.path.realpath(
            starsmashertools.helpers.path.join(
                self.src,
                starsmashertools.preferences.get_default(self, 'src init filename', throw_error=True),
            ),
        )

    def _isolate_get_input_subroutine(self, lines):
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
        return lines

    def _get_namelist_names(self, path_or_lines):
        if isinstance(path_or_lines, str):
            with starsmashertools.helpers.file.open(path_or_lines, 'r') as f:
                path_or_lines = f.read().split('\n')
        ret = []
        for line in path_or_lines:
            if 'namelist' in line and '/' in line:
                ret += [line.split('/')[1].strip()]
        return ret

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_namelist_name(
            self,
            init_file : str | type(None) = None,
    ):
        if init_file is None: init_file = self.get_init_file()
        
        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            lines = f.read().split('\n')

        lines = self._isolate_get_input_subroutine(lines)

        namelist_names = self._get_namelist_names(lines)
        if namelist_names: return namelist_names[0], init_file

        # If we got here then it means the namelist might appear in one of the
        # 'included' files
        content = '\n'.join(lines)
        for line in lines:
            if 'include' in line:
                path = line.split('include ')[1].strip().replace("'",'').replace('"','')
                path = starsmashertools.helpers.path.join(
                    starsmashertools.helpers.path.dirname(init_file),
                    path,
                )
                namelist_names = self._get_namelist_names(path)
                for name in namelist_names:
                    for line in lines:
                        if name in line and 'read(' in line:
                            return name, path
        
        raise Exception("Failed to find the input namelist name in '%s'" % init_file)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_namelist(
            self,
            init_file : str | type(None) = None,
    ):
        namelist_name, filename = self.get_namelist_name(init_file = init_file)

        with starsmashertools.helpers.file.open(filename, 'r') as f:
            lines = f.read().split('\n')

        for i, line in enumerate(lines):
            if 'namelist' in line and '/' in line:
                if line.split('/')[1].strip() == namelist_name:
                    start_idx = i
                    break
        else:
            raise Exception("This should never be possible 1")
        
        for i, line in enumerate(lines[start_idx + 1:]):
            if '$' in line or '&' in line: continue
            else:
                stop_idx = start_idx + i + 1
                break
        else:
            stop_idx = start_idx + 1

        content = ''.join(lines[start_idx:stop_idx])
        content = content.replace('$','').replace('&','')
        return [item.strip() for item in content.split('/')[2].split(',')], stop_idx
        
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

        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            lines = f.read().split('\n')

        lines = self._isolate_get_input_subroutine(lines)
        namelist_name, filename = self.get_namelist_name(init_file = init_file)
        
        for i, line in enumerate(lines):
            if 'read(' in line and namelist_name in line:
                read_index = i
                file_no = line.split('(')[1].split(',')[0]
                break
        else:
            raise Exception("Failed to find where the namelist gets read in '%s'" % init_file)

        for line in lines[:read_index][::-1]:
            if 'open' in line and file_no in line:
                for part in line.split(','):
                    if 'file' in part and '=' in part:
                        return part.split('=')[1].replace('"','').replace("'",'')
        raise Exception("Failed to find the input filename in '%s'" % init_file)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_default_values(self, init_file : str | type(None) = None):
        if init_file is None: init_file = self.get_init_file()

        input_filename = self.get_input_filename(init_file = init_file)
        
        obj = {}
        
        # Read the init.f file to obtain default values
        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            lines = f.read().split("\n")

        # Get the namelist variable names
        namelist_variables, namelist_stop_idx = self.get_namelist(init_file = init_file)
        
        # Search the remaining body of the code for variable assignments
        body = lines[namelist_stop_idx:]

        for i, line in enumerate(body):
            if line.strip().lower() in ['end', 'end subroutine', 'end subroutine get_input']:
                body = body[:i+1]
                break
        
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
                print(key, to_eval)
                obj[key] = eval(to_eval.replace("!","#"), {}, obj)
        return obj

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_input_values(
            self,
            defaults : dict | type(None) = None,
            filename : str | type(None) = None,
            init_file : str | type(None) = None,
    ):
        if init_file is None: init_file = self.get_init_file()
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
        super(Input, self).__init__(self.get_input_values())
        self._initialized = True
