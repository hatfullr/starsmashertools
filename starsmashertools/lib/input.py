import starsmashertools.helpers.string as string
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
import starsmashertools.helpers.file

# This represents the inputs sent to StarSmasher, where the keys
# are variable names and the values are the values of those variables
class Input(dict, object):
    def __init__(self, directory):
        self.directory = directory
        super(Input, self).__init__()
        self._initialized = False


    def __getitem__(self, item, **kwargs):
        if item not in list(self.keys()) and not self._initialized: self.initialize()
        return super(Input, self).__getitem__(item, **kwargs)
    
    
    def initialize(self):
        if self._initialized: raise Exception("Cannot initialize an Input object that is already initialized")
        
        filenames = preferences.get_default(self, 'input filenames', throw_error=True)

        for filename in filenames:
            inputfile = path.realpath(path.join(self.directory, filename))
            if path.isfile(inputfile): break
        else:
            raise Exception("Failed to find any input file in directory '%s'" % self.directory)


        # Find the source code directory
        src = path.get_src(self.directory, throw_error=True)
        
        init_file = path.realpath(path.join(src, preferences.get_default(self, 'src init filename', throw_error=True)))
        if not path.isfile(init_file):
            raise Exception("Missing init file '%s'" % init_file)
        
        # Read the init.f file to obtain default values
        listening = None

        with starsmashertools.helpers.file.open(init_file, 'r') as f:

            for line in f:
                # Skip empty lines
                if not line.strip(): continue

                # Always skip comments
                if line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                    continue

                if listening is None and '      subroutine get_input' in line:
                    listening = 'namelist'
                    continue

                # Get the namelist for the inputs
                if listening == 'namelist':
                    if line.strip() == "":
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
                    if line == "      end" or line.strip() == "open(12,file='sph.input',err=100)": break
                    if '=' not in line: continue

                    ls = line.strip().split('=')
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
        with starsmashertools.helpers.file.open(inputfile, 'r') as f:
            for line in f:
                ls = line.strip()

                if not ls: continue # Skip empty lines

                # Always skip comments
                if ls[0] == '!': continue

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
