import starsmashertools.helpers.string as string
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
import starsmashertools.helpers.file
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer

# This represents the inputs sent to StarSmasher, where the keys
# are variable names and the values are the values of those variables
class Input(dict, object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(self, directory : str):
        self.directory = directory
        super(Input, self).__init__()
        self._src = None
        self._initialized = False

    @property
    def src(self):
        if self._src is None:
            self._src = path.get_src(self.directory, throw_error=True)
        return self._src

    @api
    def __getitem__(self, item, **kwargs):
        if item not in list(self.keys()) and not self._initialized: self.initialize()
        return super(Input, self).__getitem__(item, **kwargs)

    def get_init_file(self):
        init_file = path.realpath(path.join(self.src, preferences.get_default(self, 'src init filename', throw_error=True)))
        if not path.isfile(init_file):
            raise Exception("Missing init file '%s'" % init_file)
        return init_file
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_input_filename(
            self,
            init_file : str | type(None) = None,
    ):

        """
        Open up the init.f file to figure out what it reads as the input file.
        We search for the 'namelist' block in the variable declarations of
        init.f and then search the simulation directory's files for one whose
        first line matches the namelist name.
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

    
    @api
    def initialize(self):
        if self._initialized: raise Exception("Cannot initialize an Input object that is already initialized")

        init_file = self.get_init_file()
        
        # Read the init.f file to obtain default values
        listening = None
        with starsmashertools.helpers.file.open(init_file, 'r') as f:
            for line in f:
                ls = line.strip()
                # Skip empty lines
                if not ls: continue

                # Always skip comments
                if line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                    continue
                
                if listening is None and line.strip().startswith('subroutine get_input'):
                    listening = 'namelist'
                    continue

                # Get the namelist for the inputs
                if listening == 'namelist':
                    if not line.strip():
                        listening = 'values'
                        continue

                    if 'namelist/input/' in line or '$' in line:
                        line = line.replace('namelist/input/','').replace('$','')
                        for name in line.strip().split(','):
                            if name:
                                self[name] = None
                    else:
                        listening = 'values'

                if listening == 'values':
                    ls = line.strip()
                    if ls.lower().startswith("end") or ls.lower().startswith("open("): break
                    if '=' not in line: continue

                    ls = ls.split('=')
                    key = ls[0].lower()

                    val = '='.join(ls[1:])
                    val = string.fortran_to_python(val)
                    self[key] = eval(val.replace("!","#"), {}, self)
        
        # Save default values
        self.defaults = {}
        for key, val in self.items():
            self.defaults[key] = val

        self.overrides = {}
        for key, val in self.items():
            self.overrides[key] = None

        # Overwrite the default values with values from the sph.input file
        inputfile = self.get_input_filename(init_file = init_file)
        inputfile = path.join(self.directory, inputfile)
        with starsmashertools.helpers.file.open(inputfile, 'r') as f:
            for line in f:
                ls = line.strip()

                if not ls: continue # Skip empty lines

                # Always skip comments
                if line[0] == '!':
                    continue

                # Only process this string if it evaluates some variable
                if '=' not in ls: continue

                ls = ls.split('=')

                key = ls[0].lower()
                val = "=".join(ls[1:])
                val = val.replace(",","")
                val = string.fortran_to_python(val)

                # Fortran cannot evaluate expressions in an input file, so we don't need to worry
                # about doing the same, but it does make it very easy to convert the string to
                # an appropriate type
                v = eval(val.replace("!","#"))
                if key in self.defaults.keys(): self.overrides[key] = v
                self[key] = v

        self._initialized = True
