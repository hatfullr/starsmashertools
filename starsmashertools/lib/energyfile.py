import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.readonlydict
import numpy as np
import collections
import mmap

class EnergyFile(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
    names = {}
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            logfile : "starsmashertools.lib.logfile.LogFile",
            skip_rows : int | type(None) = None,
    ):
        import starsmashertools.helpers.path
        import re
        import starsmashertools.helpers.warnings

        if skip_rows is None:
            skip_rows = 1000
        
        # Get the energy file path
        filename = logfile.get('writing energy data to ')
        directory = starsmashertools.helpers.path.dirname(logfile.path)
        self.path = starsmashertools.helpers.path.join(directory, filename)

        self.skip_rows = skip_rows

        if not starsmashertools.helpers.path.isfile(self.path):
            raise FileNotFoundError("'%s'. Make sure all energy*.sph files are stored directly next to the log*.sph files they correspond to." % self.path)

        if logfile.simulation.directory not in EnergyFile.names.keys():
        
            src = starsmashertools.helpers.path.get_src(logfile.simulation.directory)
            if src is None:
                raise FileNotFoundError(src)

            extensions = ['.f', '.f90']
            src_files = []
            for dirpath, dirnames, filenames in starsmashertools.helpers.path.walk(src):
                for filename in filenames:
                    _path = starsmashertools.helpers.path.join(src, filename)
                    # Skip filenames that don't end in one of the extensions listed
                    for extension in extensions:
                        if filename.endswith(extension) and starsmashertools.helpers.path.isfile(_path): break
                    else: continue

                    src_files += [_path]

            descriptor = None
            for _path in src_files:
                descriptor = EnergyFile.get_energy_file_descriptor(_path)
                if descriptor is not None: break

            if descriptor is None:
                raise Exception("Failed to find the energy file descriptor in the StarSmasher source code.")

            with starsmashertools.helpers.file.open(self.path, 'r', lock = False) as f:
                ncolumns = len(f.readline().strip().split())

            possible = []
            for _path in src_files:
                lines = EnergyFile.get_write_energy_descriptor_lines(_path, descriptor)
                # One of these lines was used to write to the energy file. We don't
                # really have a great way of figuring out which one was used. We'll
                # just take the first line that matches the number of columns in the
                # energy file.
                for line in lines:
                    if len(line.split(',')) != ncolumns: continue
                    possible += [line]

            if len(possible) == 0:
                raise Exception("Failed to parse the StarSmasher source code for the names of the variables which were written to the energy file '%s'" % self.path)

            if len(possible) > 1:
                starsmashertools.helpers.warnings.warn("Found multiple lines in the StarSmasher source code which write to the energy file the same number of columns found in the energy file. The descriptors in the energy file might not be accurate")

            EnergyFile.names[logfile.simulation.directory] = [name.strip() for name in possible[0].split(',')]
        
        data = collections.OrderedDict()
        for name in EnergyFile.names[logfile.simulation.directory]:
            data[name] = []
        
        keys = data.keys()
        with starsmashertools.helpers.file.open(self.path, 'r', lock = False) as f:
            lines = f.readlines()
        
        lines = lines[::self.skip_rows]

        for line in lines:
            for key, val in zip(keys, line.split()):
                data[key] += [val]
                
        for key, val in data.items():
            data[key] = np.asarray(val, dtype=object).astype(float)
        
        super(EnergyFile, self).__init__(data)

    def __repr__(self):
        import starsmashertools.helpers.string
        return starsmashertools.helpers.string.shorten(self.path, 25, where = 'left')

    def __str__(self): return self.__repr__()

    @staticmethod
    def normalize_file_content(content):
        import re

        expressions = [
            # Matches all comment lines that begin with "c", "C", "d", "D", "*",
            # or "\"
            r"^[cCdD*\\].*\n",
            # Matches all content after a "!" character up to the newline
            r"\!.*\n"
        ]
        
        # Remove all comments
        for expression in expressions:
            for t in re.findall(expression, content, flags = re.MULTILINE):
                content = content.replace(t, '')

        # Join lines together who are split by '$' characters or '&' characters
        expressions = [
            # Match all "$" continuations
            r"\s*\n\s*\$\s*",
            # Match all "&" continuations at end of line
            r"&\s*\n\s*",
            # Match all "&" continuations at beginning of line
            r"\n\s*&\s*",
        ]

        # Remove all line-joining characters
        for expression in expressions:
            for t in re.findall(expression, content, flags = re.MULTILINE):
                content = content.replace(t, '')
                
        return content

    @staticmethod
    def get_energy_file_descriptor(filename):
        import starsmashertools.helpers.file
        import re

        with starsmashertools.helpers.file.open(filename, 'r', lock = False) as f:
            content = f.read()

        content = EnergyFile.normalize_file_content(content)

        expression = r"^.*[wW][rR][iI][tT][eE]\(.*\)'writing energy data to ',.*[^,|\s]"
        m = re.search(expression, content, flags=re.MULTILINE)
        if not m: return None
        
        text = m.group(0)
        variable_name = text.split(',')[-1]

        # Now search for where this variable name is opened
        expression = r"^.*[oO][pP][eE][nN]\(.*[fF][iI][lL][eE].*=\s*energyfile.*"
        m = re.search(expression, content, flags=re.MULTILINE)
        if m is None: return None
        
        text = m.group(0)
        return int(text.split(',')[0].lower().replace('open','').replace('(','').strip())

    @staticmethod
    def get_write_energy_descriptor_lines(filename, descriptor):
        import starsmashertools.helpers.file
        import re

        #print(filename)
        with starsmashertools.helpers.file.open(filename, 'r', lock = False) as f:
            content = f.read()

        # Fix up the content so that it is easily parseable
        content = EnergyFile.normalize_file_content(content)

        # Find lines where energy file is being written
        expression = r"[wW][rR][iI][tT][eE]\(%d.*" % descriptor
        lines = re.findall(expression, content, flags = re.MULTILINE)
        for i, line in enumerate(lines):
            omit = re.findall(r"[wW][rR][iI][tT][eE]\(\S*\)", line, flags = re.MULTILINE)
            for o in omit:
                lines[i] = lines[i].replace(o, '')
        return lines
