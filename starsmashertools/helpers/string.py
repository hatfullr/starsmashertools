# String helper methods
import re
import starsmashertools.helpers.file
import starsmashertools.helpers.argumentenforcer

@starsmashertools.helpers.argumentenforcer.enforcetypes
def list_to_string(
        _list : list | tuple,
        join : str = 'and',
        format : str = "'%s'",
):
    if len(_list) == 0: return "''"
    if len(_list) == 1: return format % str(_list[0])
    string = ", ".join(format % s for s in _list[:-1])
    if len(_list) > 2: string += ","
    string += " " + join + " " + (format % _list[-1])
    return string

@starsmashertools.helpers.argumentenforcer.enforcetypes
def parse(string : str):
    _string = string.strip().lower()
    # Try to parse in order of ascending data complexity
    if _string in ['true', 'false']: return bool(_string)
    else:
        try:
            return int(_string)
        except ValueError as e:
            if "invalid literal for int() with base 10" not in str(e):
                raise

        try:
            return float(_string)
        except ValueError as e:
            if "could not convert string to float" not in str(e):
                raise
    # If we can't convert the string's type, just return the original string
    return string


# Convert a string read from Fortran code into a Python string,
# where the Fortran code does a variable assignment. The result
# is a string which can be passed to, e.g. 'eval', to assign a
# variable of the same name with the corresponding value from the
# Fortran code.
@starsmashertools.helpers.argumentenforcer.enforcetypes
def fortran_to_python(string : str):
    # https://regex101.com/r/wfkDtt/1
    float_match = re.compile(r"[0-9]+\.?[0-9]*[eEdD][\+|\-]?[0-9]{1,}")
    # https://regex101.com/r/0jubyV/1
    bool_match = re.compile(r"\.[tT][rR][uU][eE]\.|\.[fF][aA][lL][sS][eE]\.")
    
    for match in float_match.finditer(string):
        string = string.replace(match.string, match.string.lower().replace('d','e'))
    for match in bool_match.finditer(string):
        m = match.string.lower().replace(".","")
        m = m[0].upper() + m[1:]
        string = string.replace(match.string, m)
    return string


# Input a string of Fortran script to receive the types of each defined
# variable in that file. You must provide a list of the expected data
# types in the string, without the "*" modifier. For example, 'real'
# instead of 'real*8'. The result of this function will still return
# 'real*8' instead of 'real' if such a case is found.
@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_fortran_variable_types(
        string : str,
        data_types : list | tuple,
):
    current_type = None
    result = {}
    for line in string.split("\n"):
        if len(line) == 0 or (line[0] in starsmashertools.helpers.file.fortran_comment_characters):
            continue
        if '!' in line: line = line[:line.index('!')]
        
        line_continuation = False
        
        if len(line) >= 6:
            # https://gcc.gnu.org/onlinedocs/gcc-3.4.6/g77/Continuation-Line.html
            if line[5] not in [' ', '0']:
                line_continuation = True
        
        ls = line.strip()

        _names = None

        if line_continuation and current_type is not None:
            _names = line[6:].split(",")
        else:
            for d in data_types:
                if len(ls) > len(d) and ls[:len(d)] == d:
                    idx = ls.index(' ')
                    current_type = ls[:idx]
                    _names = ls[idx+1:].split(",")
                    break
        if _names is not None:
            for i, n in enumerate(_names):
                # Remove spaces etc.
                _names[i] = n.strip()
                # Remove array specifiers
                if '(' in _names[i]:
                    _names[i] = _names[i][:_names[i].index('(')]
            if current_type not in result.keys(): result[current_type] = _names
            else: result[current_type] += _names
        else: current_type = None
    return result
