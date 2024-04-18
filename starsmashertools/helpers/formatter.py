import starsmashertools.preferences
import starsmashertools.lib.output
import starsmashertools.lib.units
import starsmashertools.bintools.page

newline = starsmashertools.bintools.page.newline

def sheet_name_to_path(name : str):
    import starsmashertools
    import starsmashertools.helpers.path

    sheets = starsmashertools.get_format_sheets()
    for sheet in sheets:
        basename = starsmashertools.helpers.path.basename(sheet)
        if basename.replace('.format','') == name:
            return sheet
    
    raise FormatSheet.FormatSheetNotFoundError(name)




@starsmashertools.preferences.use
class FormatSheet(object):
    """
    Contains information about a format sheet.
    """

    # Valid option names
    options = {
        'unit' : None,
        'format' : None,
        'show_name' : True,
    }

    default_format = '!s:'

    class FormatSheetNotFoundError(FileNotFoundError, object): pass
    class ReadError(Exception, object): pass

    class Value(object):
        def __init__(
                self,
                x : int, y : int,
                name : str,
                options : dict = {},
        ):
            self.x = x
            self.y = y
            self.name = name
            self.options = options
    
    def __init__(
            self,
            path : str,
    ):
        import starsmashertools.helpers.path
        self.path = path
        self.name = starsmashertools.helpers.path.basename(self.path).replace('.format','')

        self.formats = {}
        self.widths = {}
        self.omit = []
        self.values = []
        
        self.read()

    def __contains__(self, name : str):
        return name in self.values.keys()
    
    def read(self):
        import starsmashertools.helpers.file
        import re
        import builtins
        import starsmashertools.helpers.string

        lines = []
        try:
            with starsmashertools.helpers.file.open(self.path, 'r', lock = False) as f:
                for line in f:
                    ls = line.strip()
                    if not ls: continue # empty line
                    if ls[0] == '#': continue # comment
                    lines += [line]
        except Exception as e:
            raise FormatSheet.ReadError() from e
            
        content = newline.join(lines)

        # Get the variables to omit
        matches = re.findall(r"omit.*=.*\[[^\]]*.", content)
        if len(matches) > 1:
            raise FormatSheet.ReadError("Cannot have more than one 'omit' statement")
        self.omit = []
        for match in matches:
            l = match.split('=')[1].replace(newline,'').strip()
            self.omit = eval(l)

        # Get the width statements
        self.widths = {
            'major' : {},
            'minor' : {},
        }
        matches = re.findall(r'.*_width.*=.*', content, flags = re.MULTILINE)
        for match in matches:
            left, right = match.split('=')
            if '_' not in left:
                raise FormatSheet.ReadError("default widths must be specified as 'name_width', not '%s'" % left)
            
            _type = left.split('_')[0].strip()
            if _type == 'default':
                l = eval(right.strip())
                self.widths['major']['default'] = l[0]
                self.widths['minor']['default'] = l[1]
                continue
            
            try:
                _type = getattr(builtins, _type)
            except AttributeError:
                raise FormatSheet.ReadError("Unrecognized builtin type '%s' in default width statement '%s'" % (_type, match))

            l = eval(right.strip())
            self.widths['major'][_type] = l[0]
            self.widths['minor'][_type] = l[1]

        # Get the default formats
        self.formats = {}
        matches = re.findall(r'.*_format.*=.*', content, flags = re.MULTILINE)
        for match in matches:
            left, right = match.split('=')
            _type = left.split('_')[0].strip()
            try:
                self.formats[getattr(builtins, _type)] = right.strip()
            except AttributeError:
                raise FormatSheet.ReadError("Unrecognized builtin type '%s' in default format statement '%s'" % (_type, match))

        
        # Fix up the formats using the widths
        major_keys = self.widths['major'].keys()
        minor_keys = self.widths['minor'].keys()
        for key, val in self.formats.items():
            if not ('W' in val or 'w' in val): continue
            if key not in major_keys and key not in minor_keys:
                raise FormatSheet.ReadError("Cannot specify 'W' and/or 'w' in a format for which there is no default width given, for type '%s'" % str(key))
            
            major = self.widths['major'][key]
            minor = self.widths['minor'][key]
            v = val.replace('W', str(major)).replace('w', str(minor))
            self.formats[key] = v
            
        # Get the values
        self.values = {}
        position_re = r'^\(.*\d+.*,.*\d+.*\)'
        matches = re.findall(
            r'^\(.*\d+.*,.*\d+.*\)[^\S\r\n]*\S*[^\S\r\n]*.*',
            content,
            flags = re.MULTILINE,
        )
        for match in matches:
            position_string = re.findall(position_re, match)[0]
            s = position_string.replace('(','').replace(')','').strip()
            x, y = [int(t.strip()) for t in s.split(',')]

            match = match[len(position_string) + 1:]
            name = match.split(' ')[0].strip()

            match = match[len(name) + 1:]
            options = {}
            for k,v in FormatSheet.options.items():
                options[k] = v

            if match:
                for m in match.split(','):
                    opname, opval = m.split('=')
                    opname = opname.strip()
                    opval = opval.strip()

                    if opname not in options.keys():
                        raise FormatSheet.ReadError("Unrecognized option '%s'" % opname)

                    try:
                        opval = getattr(builtins, opval)
                    except AttributeError:
                        opval = starsmashertools.helpers.string.parse(opval)

                    options[opname.strip()] = opval
            
            self.values[name] = FormatSheet.Value(
                x, y, name,
                options = options,
            )

        #print(self.formats)

    def get_format(self, name : str, val):
        import copy
        import numpy as np

        # The default format
        fmt = self.formats.get(
            'default',
            FormatSheet.default_format,
        )
        
        # Convert NumPy types
        if isinstance(val, np.generic): val = val.item()
        
        if name not in self:
            fmt = self.formats.get(type(val), fmt)
            return fmt
        
        _type = type(val)
        if isinstance(val, starsmashertools.lib.units.Unit):
            _type = float
        
        value = self.values[name]

        optfmt = value.options.get('format', None)
        if optfmt is not None:
            fmt = self.formats.get(optfmt, fmt)
        else:
            fmt = self.formats.get(_type, fmt)
        
        return fmt

    def get_unit(self, name : str):
        if name not in self: return None
        return self.values[name].options['unit']


class Formatter(object):
    """
    A helper class for using format sheets to produce outputs.
    """
    def __init__(
            self,
            sheet : str | FormatSheet,
    ):
        try:
            self.sheet = FormatSheet(sheet)
        except FormatSheet.ReadError:
            self.sheet = FormatSheet(sheet_name_to_path(sheet))
    
    def __repr__(self): return 'Formatter(%s)' % self.sheet.name

    def get_string(
            self,
            name : str,
            value : float | int | bool | str | starsmashertools.lib.units.Unit,
            namelength : int | type(None) = None,
    ):
        unit_label = ''
        if isinstance(value, starsmashertools.lib.units.Unit):
            unit = self.sheet.get_unit(name)
            if unit is not None: value = value.convert(unit)
            else: value = value.auto()
            unit_label = value.label
            value = float(value)
        
        v = self.sheet.values.get(name, None)
        fmt = ''
        if v is not None:
            if v.options['show_name']: fmt = '{name} = '
        else:
            if FormatSheet.options['show_name']:
                fmt = '{name} = '

        if namelength is not None:
            fmt = fmt.replace('{name}', '{name:>'+str(namelength)+'s}')
                
        fmt += self.sheet.get_format(name, value)
        if unit_label: fmt += ' {unit_label}'
        return fmt.format(
            value,
            name = name,
            unit_label = unit_label,
        )
    
    def get_extent(
            self,
            values : dict,
            namelength : int | type(None) = None,
    ):
        """ 
        Get the total character width the sheet would have if given values.
        """
        xmin, xmax, ymin, ymax = None, None, None, None
        for name, val in values.items():
            if name not in self.sheet: continue
            value = self.sheet.values[name]
            string = self.get_string(name, val, namelength=namelength)
            x0 = value.x
            x1 = x0 + len(string)
            y = value.y
            if xmin is None: xmin = x0
            if xmax is None: xmax = x1
            if ymin is None: ymin = y
            if ymax is None: ymax = y
            xmin, xmax = min(xmin, x0), max(xmax, x1)
            ymin, ymax = min(ymin, y), max(ymax, y)
        if xmin is None: xmin = 0
        if xmax is None: xmax = 0
        if ymin is None: ymin = 0
        if ymax is None: ymax = 0
        # At least nonzero width and height
        xmax = max(xmax, 1)
        ymax = max(ymax, 1)
        return [xmin, xmax, ymin, ymax]

    def format_output(
            self,
            output : starsmashertools.lib.output.Output,
    ):
        """ Obtain the formatted string using the format sheet. """
        units = output.simulation.units
        keys = []
        values = []
        for key, val in output.header.items():
            keys += [key]
            if key in units.keys():
                values += [val * units[key]]
            else:
                values += [val]

        # Sort the keys and values by keys (alphabetical)
        values = [val for key, val in sorted(zip(keys, values))]
        keys = sorted(keys)

        # Normalize the name lengths to all be the same
        maxlen = max([len(k) for k in keys])
        namelength = max([len(k.rjust(maxlen)) for k in keys])

        extent = self.get_extent({k:v for k,v in zip(keys, values)}, namelength=namelength)
        width = extent[1] - extent[0] + 1
        height = extent[3] - extent[2] + 1

        # Create the main string
        arr = [[" " for i in range(width)] for j in range(height)]

        # Add all the pre-defined values first
        remaining = []
        for key, val in zip(keys, values):
            if key not in self.sheet:
                remaining += [[key, val]]
                continue
            v = self.sheet.values[key]
            string = self.get_string(key, val, namelength=namelength)
            
            for i, char in enumerate(list(string)):
                arr[v.y][v.x + i] = char

        # Fill in the rest of the undefined values
        arr += [""] # Newline
        for key, val in remaining:
            string = self.get_string(key, val, namelength = namelength)
            arr += [string]

        arr = [str(output)] + ["".join(a) for a in arr]
        return newline.join(arr)
