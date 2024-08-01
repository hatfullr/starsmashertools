import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import re
import numpy as np

variable_statements = [
    'integer',
    'real',
    'double precision',
    'complex',
    'double complex',
    'logical',
    'character',
    'byte',
]

object_statements = [
    'function',
    'subroutine',
]

nonexecutable_statements = [
    'program',
    'module',
    'entry',
    'block data',
    'dimension',
    'common',
    'equivalence',
    'implicit',
    'parameter',
    'external',
    'intrinsic',
    'save',
    'integer',
    'data',
    'format',
] + object_statements + variable_statements

# Refer to these pages:
# https://docs.oracle.com/cd/E19957-01/805-4939/c400041360f5/index.html
# https://numpy.org/doc/stable/reference/arrays.dtypes.html ("Array-protocol type strings")
numpy_formats = {
    'byte' : {
        1 : 'B1',
    },
    'complex' : {
        8 : 'c8',
        16 : 'c16',
        32 : 'c32',
    },
    'double complex' : {
        16 : 'c16',
    },
    'double precision' : {
        8 : 'f8',
    },
    'real' : {
        4 : 'f4',
        8 : 'f8',
        16 : 'f16',
    },
    'integer' : {
        2 : 'i2',
        4 : 'i4',
        8 : 'i8',
    },
    'logical' : {
        1 : '?1',
        2 : '?2',
        4 : '?4',
        8 : '?8',
    },
}

# Some exceptions
class EndianError(Exception): pass
class ParsingError(Exception): pass


@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_numpy_format(
        record : bytes,
        variables : list | tuple,
        endian : str | type(None) = None,
):
    r"""
    Return a format string that can be used in :class:`numpy.ndarray` for the 
    provided bytes of data that originated from a Fortran file.

    Parameters
    ----------
    record : bytes
        The bytes to check the variables against.

    Other Parameters
    ----------------
    endian : str, None, default = None
        If the endianness of ``record`` is known, a ``'<'`` or ``'>'`` is
        appended to the start of the string where ``'<'`` is for little endian
        and ``'>'`` is for big endian. If the endianness is not known, 
        ``endian`` should be given as `None`\, and the endianness will be 
        determined using the start-of-record and end-of-record markers provided
        in ``record`` as written by Fortran. These are 4 byte integers which
        indicate the length of the record.

    Returns
    -------
    fmt : str, None
        A valid format string that can be used in :class:`numpy.ndarray` for the
        given ``record``\. If `None`\, then the given ``record`` is incompatible
        with the given ``variables``\.

        If any ``real`` or ``double precision`` variables have a value for which
        the magnitude of its exponent exceeds 100, returns `None`\. Usually junk
        values take on the form of having unrealistically large or small 
        exponents, like +300 or -300. There doesn't exist any good way to detect
        bad values like this, so we instead just check for exponents that exceed
        100 in magnitude. That is, if the exponent is less than -100 or greater
        than 100, we skip this write statement.

    Raises
    ------
    :class:`~.EndianError`
        If the provided or detected endian is not ``'little'`` or ``'big'``\.
    """
    if sum([v.size for v in variables]) != len(record) - 8: return None
    if endian is None: endian = get_endian(record)

    fmt = ','.join([v.get_numpy_format() for v in variables])
    if endian == 'little': fmt = '<' + fmt
    elif endian == 'big': fmt = '>' + fmt
    else: raise EndianError("Unrecognized endian '%s'" % str(endian))
    
    result = np.ndarray(
        buffer = record,
        shape = 1, # Reading a single line
        dtype = fmt,
        offset = 4,
        strides = len(record)-8,
    )[0]

    for variable, res in zip(variables, result):
        if variable.kind in ['real','double precision']:
            # Check if the exponent of the number exceeds 100. Usually junk
            # values take on the form of having unrealistically large or small
            # exponents, like +300 or -300. There doesn't exist any good way to
            # detect bad values like this, so we instead just check for
            # exponents that exceed 100 in magnitude. That is, if the exponent
            # is less than -100 or greater than 100, we skip this write
            # statement.
            if res != 0 and abs(np.log10(abs(res))) > 100:
                return None
    return fmt


@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def parse_write_statement(string : str):
    r"""
    Given a stripped Fortran write statement, return the variable names. If the
    write statement includes a string, returns an empty list.
    
    Returns
    -------
    names : str
        The names of each variable in the write statement.
    """

    rules = [
        # Find the write statement if it has a Fortran formatting string
        # https://regex101.com/r/lN3i2u/2
        re.compile(r"[wW][rR][iI][tT][eE] *\(.*?(?<!\\)([\"']).*?(?<!\\)\1\)"),
        # If the previous rule wasn't matched, try this one instead
        # https://regex101.com/r/qo5l2T/1
        re.compile(r"[wW][rR][iI][tT][eE] *\(.*?\)"),
    ]
    write = None
    for rule in rules:
        matches = [x.group(0) for x in rule.finditer(string)]
        if not matches: continue
        write = matches[0]
        break
    if write is None:
        raise ParsingError("Failed to recognize the 'write' Fortran directive in write statement: '%s'" % string)

    # Throw out the "write" fortran directive and focus only on the variables
    string = string[string.index(write)+len(write):]

    for c in ['"', "'"]:
        if c in string:
            raise ParsingError("Cannot parse a write statement that contains string literals: %s" % string)
    for c in ['+','-','/','*']:
        if c in string:
            raise ParsingError("Cannot parse a write statement that contains math operators +, -, /, or *: '%s'" % string)

    lp, rp = 0, 0
    name = ''
    names = []
    for i, c in enumerate(string):
        if c == ' ': continue # skip white space
        elif c == '(':
            lp += 1
            continue
        elif c == ')':
            rp += 1
            continue
        
        if lp != rp: # In parentheses
            continue
        
        if c == ',': # the variables delimiter
            if lp == rp: # Not in parentheses
                names += [name]
                name = ''
                continue
        else: name += c
    if name: names += [name]
    return names


@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def strip(source : str):
    r"""
    Given the contents of a Fortran file, this method returns the file contents
    in a more consistent format. The Fortran comments are removed, continuation
    lines are merged into single lines, and all non-newline whitespaces are 
    removed.
    
    Parameters
    ----------
    source : str, :py:class:`io.TextIOBase`
        If a :py:class:`str`\, it is expected to be a file path. Otherwise, it
        is expected to be an opened file (in text format).
    
    Returns
    -------
    contents : str
        The contents of the file without any Fortran comments.

    Notes
    -----
    Sun Fortran is not supported, so tabs in the source code are not expected
    and will likely cause errors.

    Source code containing semicolons as statement separators is not supported.
    """
    import starsmashertools.helpers.string
    
    def strip_empty_lines(lines):
        for line in lines:
            if not line.strip(): continue
            yield line

    def strip_comments(lines):
        for line in lines:
            if any([line.startswith(c) for c in ['c','C','!','*','d','D']]):
                continue
            yield starsmashertools.helpers.string.strip_inline_comment(
                line, '!',
            )
    
    def merge_continuations(lines):
        # Assuming that comments have already been removed and whitespace has
        # not yet been removed
        lines = '\n'.join(lines)

        # The regex101.com examples provided correspond with the unit test in
        # tests/test_file.py
        regs = [
            # (F90+) Lines that end with & with 6 spaces in the next line. Here
            # we match 6 or more spaces so that the whitespace is removed from
            # the continuation line. We replace the space padding with just a
            # single space instead.
            # 3.3.1.3.1 Noncharacter context continuation
            # https://regex101.com/r/o7tiZg/1
            {'args':(r' +&.*\n +& +',' '),'kwargs':{'flags':re.M}},
            # If a lexical token is split across the end of a line, the first
            # nonblank character on the following line must be an "&"
            # https://regex101.com/r/itKtOV/1
            {'args':(r' *\n +& +', ' '),'kwargs':{'flags':re.M}},
            # 3.3.1.3.2 Character context continuation
            # https://regex101.com/r/nwqz80/2
            {'args':(r' +&.*\n {6,}',' '),'kwargs':{'flags':re.M}},

            # When an & ends a line and it wasn't after a blank space, then we
            # don't need to replace that blank space. We do need to replace the
            # 6 spaces at the start of the continuation line, though.
            # https://regex101.com/r/bEeJiq/2
            {'args':(r'&.*\n {6}', ''),'kwargs':{'flags':re.M}},
            
            
            # (F77) Continuation lines have only spaces in columns 1-5 and
            # contains a continuation character in column 6, which is anything
            # other than space or 0.
            # https://regex101.com/r/qNrsSn/3
            {'args':(r' *\n {5}[^ 0] *',' '), 'kwargs':{'flags':re.M}},
            # A continuation line can also start with a digit 0-9 in columns
            # 7+, provided it is preceeded by only spaces.
            # https://regex101.com/r/MzjGuc/5
            {'args':(r' *\n {6,}[0-9]',''), 'kwargs':{'flags':re.M}},
        ]

        for reg in regs:
            lines = re.sub(*reg['args'], lines, **reg['kwargs'])
        return lines.split('\n')
    
    lines = merge_continuations(
        strip_empty_lines(
            strip_comments(source.split('\n'))
        )
    )
    # Remove columns 1-6, check for tabs, and remove white space
    return re.sub(
        # https://regex101.com/r/gMkglu/1
        r'^[ 0-9]{,6}[ \t]*|[ \t]+$',
        '',
        '\n'.join(lines),
        flags = re.M,
    )

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def is_stripped(string : str):
    r"""
    Returns
    -------
    is_stripped : bool
        `True` if Fortran code in ``string`` is stripped by :func:`~.strip`\, or
        `False` otherwise.
    """
    
    if len(string) > 6:
        return string[6:] == strip(string)
    # columns 1-6 are removed by the strip function
    return False

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_endian(record : bytes):
    r"""
    Returns 'little' if the given Fortran record is little endian and 'big' if 
    it is big endian. The endianness is checked by finding the first record 
    marker, which is the first 4 bytes of the record, and attempting to convert
    that to an integer. If that length exceeds the actual length of the record,
    an error is raised. Otherwise, the 4 bytes proceeding the record length is
    read. If the two record markers match, then the endianness of the record is
    returned.
    
    Parameters
    ----------
    record : bytes
        The Fortran record to determine the endianness for. It must contain the
        leading and trailing record markers.

    Returns
    -------
    endian : str
        Either ``'little'`` or ``'big'``\.

    Raises
    ------
    :class:`~.EndianError`
        When the record length is less than 8 bytes, the first 4 bytes of the
        record do not match the last 4 bytes, or if the endianness was neither
        little nor big (should never happen).

    See Also
    --------
    :func:`~.get_endian_from_file`

    Notes
    -----
    https://gcc.gnu.org/onlinedocs/gfortran/File-format-of-unformatted-sequential-files.html
    """
    if len(record) < 8:
        raise EndianError("The length of the given record must be at least greater than or equal to the length of two record markers (4 bytes each): %s" % str(record))
    if record[:4] != record[-4:]:
        raise EndianError("The first 4 bytes of the given record must be identical to the last 4 bytes, as these are the record markers: %s" % str(record))
    
    for endian in ['little', 'big']: # little endian is more common, I think
        recl = int.from_bytes(record[:4], byteorder = endian)
        if recl == len(record) - 8: return endian
    
    raise EndianError("Failed to detect endianness of record %s" % str(record))


def get_endian_from_file(path : str):
    r"""
    Similar to :func:`~.get_endian`\, except operates on a file instead.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    endian : str
        Either ``'little'`` or ``'big'``\.

    Raises
    ------
    :class:`~.EndianError`
        When the record length is less than 8 bytes, the first 4 bytes of the
        record do not match the last 4 bytes, or if the endianness was neither
        little nor big (should never happen).

    See Also
    --------
    :func:`~.get_endian`

    Notes
    -----
    https://gcc.gnu.org/onlinedocs/gfortran/File-format-of-unformatted-sequential-files.html
    """
    import starsmashertools.helpers.file

    with starsmashertools.helpers.file.open(
        path, 'rb', lock = False, verbose = False,
    ) as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)

        if size < 8: raise EndianError("File size cannot be less than 8 bytes, as the length of a record marker is 4 bytes: '%s'" % path)
        
        record_marker = f.read(4)
        recl_little = int.from_bytes(record_marker, byteorder = 'little')
        if recl_little + 8 > size: return 'big'
        f.seek(4 + recl_little)
        if recl_little == int.from_bytes(f.read(4), byteorder = 'little'):
            return 'little'
        else: return 'big'

@starsmashertools.preferences.use
class FortranFile(object):
    class FileExtensionError(Exception): pass
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(self, path : str):
        r"""
        A file containing Fortran source code.

        Parameters
        ----------
        path : str
            The path to the Fortran file.

        Notes
        -----
        Sun Fortran is not supported, so tabs in the source code will likely 
        cause errors.

        Source code containing semicolons as statement separators is not 
        supported.

        Statement functions are not supported and will lead to undefined 
        behaviors. Read more about these here:
        https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vnb9/index.html

        Derived data types are not recognized.
        """
        import starsmashertools.helpers.string
        for ext in FortranFile.preferences.get('file extensions'):
            if path.endswith(ext): break
        else: raise FortranFile.FileExtensionError("Fortran file does not end with one of the valid extensions (%s): %s. You can set the valid Fortran file extensions in your preferences." % (
                starsmashertools.helpers.string.list_to_string(
                    FortranFile.preferences.get('file extensions'),
                    join = 'or',
                ),
                path
        ))
        self.path = path
        self._content = None
        self.get_contents()

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_contents(self, stripped : bool = True):
        r"""
        Get the contents of the file.

        Other Parameters
        ----------------
        stripped : bool, default = True
            If `False`\, the returned contents are unmodified. If `True`\, the
            returned contents has comments removed, continuation lines merged,
            columns 1-6 removed, and each line is stripped of leading and
            trailing white space.

        Returns
        -------
        content : str
        """
        if self._content is None:
            import starsmashertools.helpers.file
            with starsmashertools.helpers.file.open(
                    self.path, 'r', lock = False, verbose = False,
            ) as f:
                self._content = f.read()
        if stripped: return strip(self._content)
        return self._content

    @api
    def get_objects(self):
        r"""
        Get the functions and subroutines in the file.

        Yields
        ------
        object : :class:`~.Object`
        """
        object_content = None

        contents = self.get_contents(stripped = True)
        for kind in ['function', 'subroutine']:
            for match in re.findall(
                    # https://regex101.com/r/vGfZVE/1
                    r'^ *%s[\s\S]*?^ *[eE][nN][dD]$|^ *[eE][nN][dD] +%s' % (kind, kind),
                    contents,
                    flags = re.M,
            ):
                yield Object(match, self.path)

    @api
    def get_write_statements(self):
        r"""
        Find all lines which involve the ``write`` Fortran directive and obtain
        the written variables. Write statements which include strings are 
        ignored.
        
        Yields
        ------
        statement : str
            The stripped lines which contain write statements.

        variables : list
            A list of :class:`~.Variable` objects involved in the write 
            statement, in the order they are written.
        """
        for obj in self.get_objects():
            variables = {var.name:var for var in obj.variables}
            for statement in re.findall(
                    # https://regex101.com/r/yiqSre/4
                    r"[wW][rR][iI][tT][eE] *\([^\)]+\) *[^\n]*$",
                    obj.body,
                    flags = re.M,
            ):
                # Every variable name must be in our list of variables.
                # Otherwise, the write statement probably contains either string
                # literals, operators, or function calls, none of which we can
                # reasonably parse without huge effort.
                try:
                    names = parse_write_statement(statement)
                except ParsingError as e:
                    if ('contains string literals' in str(e) or
                        'math operators' in str(e)): continue
                    raise(e)
                for name in names:
                    if name not in variables: break
                else:
                    yield statement, [variables[name] for name in names]
        

class Variable(object):
    def __init__(
            self,
            name : str,
            kind : str,
            size : int,
    ):
        r"""
        A Fortran variable.

        Paramters
        ---------
        name : str
            The name of the variable.

        kind : str
            The type of variable, such as ``real``\, ``integer``\, etc.

        size : int
            The size of the variable in bytes.
        """
        self.name = name
        self.kind = kind
        self.size = size
    def __repr__(self): return 'Variable(%s, %s)' % (self.name, self.kind)

    def __eq__(self, other):
        if not isinstance(other, Variable): return False
        return self.name == other.name and self.kind == other.kind

    def get_numpy_format(self):
        if self.kind not in numpy_formats:
            raise KeyError("Fortran variable has no corresponding NumPy data type: %s" % str(self))
        fmts = numpy_formats[self.kind]
        if self.size not in fmts:
            raise KeyError("Fortran variable has an unimplemented size %d for its NumPy data type: %s" % (self.size, str(self)))
        return fmts[self.size]
    
    def get_value(self, obj : bytes, endian : str):
        r""" Get the value of this variable from the given byte string. """
        if endian == 'little': endian = '<'
        elif endian == 'big': endian = '>'
        return np.ndarray(
            buffer = obj,
            shape = 1,
            dtype = endian + self.get_numpy_format(),
            strides = self.size,
        )[0]

    @staticmethod
    def from_string(string : str):
        kind = None
        for s in variable_statements:
            if not string.lower().startswith(s): continue
            kind = s
            break
        else: return

        if kind == 'character': return

        # Get the size specifier of the variable, if there is one
        matches = re.findall(r'\*[0-9]+', string)
        if matches:
            size = int(matches[0][1:])
        else:
            # Default variable sizes from
            # https://docs.oracle.com/cd/E19957-01/805-4939/c400041360f5/index.html
            if kind == 'integer': size = 4
            elif kind == 'real': size = 4
            elif kind == 'double precision': size = 8
            elif kind == 'logical' : size = 4
            elif kind == 'byte': size = 1
            elif kind == 'complex': size = 8
            else: raise NotImplementedError("Fortran variable type '%s' not recognized" % kind)

        string = string[len(kind):]
        if '::' in string: string = string[string.index('::')+len('::'):]

        vars_without_sizes = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_$]*(?![^\(,\n* /])(?! *\*[0-9]+)',
            string,
        )
        vars_with_sizes = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_$]*(?![^\(,\n* /]) *\*[0-9]+',
            string,
        )

        for match in vars_without_sizes:
            yield Variable(match, kind, size)
        for match in vars_with_sizes:
            idx = match.index('*')
            size = int(match[idx+1:].strip())
            yield Variable(match[:idx].strip(), kind, size)
        

    @staticmethod
    def from_include_header(path : str):
        import starsmashertools.helpers.file

        with starsmashertools.helpers.file.open(
            path, 'r', lock = False, verbose = False,
        ) as f:
            content = f.read()
        content = strip(content)
        for line in content.split('\n'):
            for s in variable_statements:
                if not line.lower().startswith(s): continue
                yield from Variable.from_string(line)
                break


class Object(object):
    r"""
    A base class for Fortran objects such as functions and subroutines. Each
    object contains variable declarations at its start. The provided content
    must first be stripped using :meth:`~.strip`\.
    """

    kinds = [
        'function',
        'subroutine',
    ]
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(self, content : str, file_path : str):
        import starsmashertools.helpers.path

        self.file_path = file_path
        
        # The last line is always "end" or "end subroutine", etc.
        lines = content.split('\n')

        # The first line of the content is always "subroutine ..." or
        # "function ..." etc.
        self.kind = re.findall(
            r'[a-zA-Z_][a-zA-Z0-9_$]*(?![^(]*\))',
            lines[0],
            flags = re.M,
        )[0]
        
        n = lines[0][lines[0].index(' ')+1:]
        if '(' in n:
            self.name = n[:n.index('(')]
        else:
            self.name = n
        
        lines = lines[1:-1]
        
        # Variable declarations stop at the first executable statement
        header = []
        body = []
        for line in lines:
            if not (any([line.startswith(s) for s in nonexecutable_statements]) or
                line.lower().startswith('include ') or
                line.lower().startswith('use ')):
                break
            header += [line]
        
        for line in lines:
            if line in header: continue
            if line.lower().startswith('end '): break
            body += [line]
            
        self.body = '\n'.join(body)
        self.header = '\n'.join(header)
        
        self._variables = None

    @property
    def variables(self):
        if self._variables is None:
            import starsmashertools.helpers.path
            
            # Detect variables declared in the declarations
            self._variables = []
            for line in self.header.split('\n'):
                # include statements are handled below
                if line.strip().lower().startswith('include '): continue
                self._variables += list(Variable.from_string(line))

            # Detect variables that come from 'include' statements
            # Here we get the relative paths to the header files
            for match in re.findall(
                    r"(?<=[iI][nN][cC][lL][uU][dD][eE] ')[^']*(?=')|(?<=[iI][nN][cC][lL][uU][dD][eE] \")[^\"]*(?=\")",
                    self.header,
                    flags = re.M
            ):
                p = starsmashertools.helpers.path.join(
                    starsmashertools.helpers.path.dirname(self.file_path),
                    match,
                )
                if not starsmashertools.helpers.path.exists(p): continue
                self._variables += list(Variable.from_include_header(p))
        return self._variables
        
    @property
    def is_function(self): return self.kind == 'function'
    @property
    def is_subroutine(self): return self.kind == 'subroutine'
