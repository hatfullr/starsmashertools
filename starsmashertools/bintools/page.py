# A Page is an object which displays information in the terminal and is able to
# navigate to other Page objects.
import sys
import starsmashertools.bintools
import starsmashertools.bintools.inputmanager
import copy

newline = starsmashertools.bintools.Style.get('characters', 'newline')

class Page(object):
    def __init__(
            self,
            cli,
            inputtypes,
            contents,
            header=None,
            footer=None,
            identifier=None,
            inputmanager=None,
            indent=1,
            back=None,
            _quit=True,
    ):
        super(Page, self).__init__()
        self.cli = cli
        self.inputtypes = inputtypes
        self.contents = contents
        self.header = header
        self.footer = footer
        self.identifier = identifier
        self.inputmanager = inputmanager
        self.indent = indent
        self.triggers = {}
        self._prompt_kwargs = {}
        self._quit = _quit

        self._back = None
        self._back_asprompt = False
        if back is not None: self.add_back(back, asprompt=True)

    def prompt(self, **kwargs):
        if not kwargs: kwargs = self._prompt_kwargs
        if self.inputmanager is None:
            raise Exception("No InputManager was given for this page")
        return self.inputmanager.get(self.inputtypes, **kwargs)

    def process_input(self, _input):
        if _input in self.triggers.keys():
            self.triggers[_input]()
            return True
        if self._back is not None and _input in ['b', 'back']:
            self.cli.navigate(self._back.identifier)
        if self._quit and _input in ['q', 'quit']:
            quit()
        return False

    # Keywords go to the input manager
    def show(self, skip=False, back=None, _quit=None, **kwargs):
        if callable(self.contents): content = self.contents()
        else: content = copy.copy(self.contents)
        
        content = newline.join([" "*self.indent + c for c in content.split(newline)])
        
        header = copy.copy(self.header)
        footer = copy.copy(self.footer)
        if header is None: header = starsmashertools.bintools.Style.get('formatting', 'header')
        if footer is None: footer = starsmashertools.bintools.Style.get('formatting', 'footer')
        
        if header: content = header + content
        if footer: content += footer

        content = newline.join([" "+c for c in content.split(newline)])
        
        self.cli.write(content, flush=True)
        
        if skip: return

        content = []
        if back is None:
            if self._back_asprompt and self._back is not None:
                content += ["b) back"]
        elif back:
            content += ["b) back"]

        if _quit is None:
            if self._quit:
                content += ["q) quit"]
        elif _quit:
            content += ["q) quit"]
        
        if content:
            content = newline.join(content)
            self.cli.write(content, flush=True)
        
        if self.inputmanager is not None and self.inputtypes:
            self._prompt_kwargs = kwargs
            if self.process_input(self.prompt()): return

        if not self._back_asprompt and self._back is not None:
            self._back.show()
            return
            
        self._on_no_connection()

    def _on_no_connection(self):
        self.cli.reset()
            

    # Connecting two pages informs the CLI how to navigate the pages. The
    # 'triggers' argument should be an iterable whose elements are possible user
    # inputs.
    def connect(self, other, triggers):
        if not hasattr(triggers, '__iter__') or isinstance(triggers, str):
            raise TypeError("Argument 'triggers' must be a non-str iterable")
        if not isinstance(other, Page): # It's an identifier
            other = self.cli.pages[other]
        for trigger in triggers:
            if trigger in self.triggers.keys():
                raise KeyError("Trigger already exists: '%s'" % str(trigger))
            self.triggers[trigger] = lambda: self.cli.navigate(other.identifier)
    

    def add_back(self, page, asprompt=True):
        self._back = page
        self._back_asprompt = asprompt
        if self._back_asprompt:
            if str not in self.inputtypes:
                self.inputtypes += [str]

    def add_quit(self):
        self._quit = True
        if str not in self.inputtypes:
            self.inputtypes += [str]


class List(Page, object):
    def __init__(self, cli, inputtypes, items=[], bullet="%d)", separator=' ', **kwargs):
        super(List, self).__init__(cli, inputtypes, "", **kwargs)
        self.items = []
        self.bullet = bullet
        self.separator = separator
        for item in items: self.add(item)

    def show(self, *args, **kwargs):
        self.contents = []
        for i, item in enumerate(self.items):
            if item.text:
                if item.bullet is None: item.bullet = self.bullet % i
                if item.separator is None: item.separator = self.separator
            self.contents += [item.to_string()]
        self.contents = newline.join(self.contents)
        super(List, self).show(*args, **kwargs)
        
    def add(self, string, **kwargs):
        self.items += [List.Item(string, **kwargs)]

    def _on_no_connection(self):
        error = starsmashertools.bintools.inputmanager.InputManager.InvalidInputError("Invalid list selection")
        starsmashertools.bintools.print_error(error=error)
        result = self.process_input(self.prompt())
        if result: return
        else: self._on_no_connection()


    class Item(object):
        def __init__(self, text, bullet=None, separator=None):
            self.text = text
            self.bullet = bullet
            self.separator = separator

        def to_string(self):
            string = ''
            if self.bullet: string += self.bullet
            if self.separator: string += self.separator
            return string + self.text




class Table(Page, object):
    """Display a table of data in the terminal. The content given as input must
    be either a list of columns or a callable function that takes no inputs and
    returns a list of columns. Columns themselves should be lists, such that the
    content appears as, e.g. ``[[0,1],[0,1],[2,3]]``, which would give a table
    like:
        ``0 0 2``
        ``1 1 3``
    The labels should also be either a list of lists or a callable function. You
    can alternatively specify your columns as, e.g.
    ``[[[0,1],[0,1],[2,3]],['col1','col2','col3']]``, where 'col1', 'col2', and
    'col3' are the column labels:
        ``col1 col2 col3``
        ``   0    0    2``
        ``   1    1    3``
    Likewise, your columns function can return a similar object. Columns and
    labels are evaluated at runtime during the show() function.
    """
    def __init__(self, cli, inputtypes, columns, labels=None, **kwargs):
        self.columns, self.labels = self.parse_columns_and_labels(
            columns,
            labels=labels,
        )
        
        super(Table, self).__init__(cli, inputtypes, "", **kwargs)

    def parse_columns_and_labels(self, columns, labels=None):
        if callable(columns): return columns, labels
        if (len(columns) == 2 and
            isinstance(columns[0], list) and
            isinstance(columns[1], list) and
            isinstance(columns[0][0], list)):
            if labels is not None:
                raise ValueError("Keyword argument 'labels' must be 'None' when labels are provided in argument 'columns', but received: '%s'" % str(labels))
            return columns[0], columns[1]
        return columns, labels

    def get_column_widths(self, columns):
        # Determine column widths
        widths = []
        for column in columns:
            max_width = 0
            if isinstance(column, str):
                max_width = len(starsmashertools.bintools.Style.clean(column))
            else:
                for element in column:
                    max_width = max(max_width, len(starsmashertools.bintools.Style.clean(element)))
            widths += [max_width]
        return widths

    def show(self, *args, **kwargs):
        if callable(self.columns):
            self.columns, self.labels = self.parse_columns_and_labels(
                self.columns(),
                labels=self.labels if not callable(self.labels) else self.labels(),
            )
        if callable(self.labels): self.labels = self.labels()

        if self.labels is None: self.labels = [""]*len(self.columns)
        
        if len(self.labels) != len(self.columns):
            raise Exception("The number of labels and columns are mismatched")
        
        # Determine the contents
        v = starsmashertools.bintools.Style.get('characters', 'table vertical')
        h = starsmashertools.bintools.Style.get('characters', 'table horizontal')
        tl = starsmashertools.bintools.Style.get('characters', 'table top left')
        tr = starsmashertools.bintools.Style.get('characters', 'table top right')
        bl = starsmashertools.bintools.Style.get('characters', 'table bottom left')
        br = starsmashertools.bintools.Style.get('characters', 'table bottom right')
        ts = starsmashertools.bintools.Style.get('characters', 'table top separator')
        bs = starsmashertools.bintools.Style.get('characters', 'table bottom separator')
        ls = starsmashertools.bintools.Style.get('characters', 'table left separator')
        rs = starsmashertools.bintools.Style.get('characters', 'table right separator')
        fs = starsmashertools.bintools.Style.get('characters', 'table full separator')
        
        column_widths = self.get_column_widths(self.columns)
        
        
        if self.labels:
            label_widths = self.get_column_widths(self.labels)
            column_widths = [max(c, l) for c, l in zip(column_widths, label_widths)]
        widths = column_widths

        self.contents += tl + ts.join([h*width for width in widths]) + tr + newline

        formatters = []
        if self.labels:
            labels_to_use = []
            for label, width in zip(self.labels, widths):
                formatter, string = starsmashertools.bintools.Style.get_formatter(label, "{:>" + str(width) + "s}")
                formatters += [formatter]
                labels_to_use += [string]
            formatter = "{left}" + "{separator}".join(formatters) + "{right}" + newline
            self.contents += formatter.format(*labels_to_use, left=v, right=v, separator=v)
            self.contents += formatter.format(*[h*width for width in widths], left=ls, right=rs, separator=fs)

        max_col_len = 0
        column_formatters = []
        column_contents = []
        for column, width in zip(self.contents, widths):
            formatters = []
            to_use = []
            for element in column:
                formatter, string = starsmashertools.bintools.Style.get_formatter(element, "{:>" + str(width) + "s}")
                formatters += [formatter]
                to_use += [string]
            column_formatters += [formatters]
            column_contents += [to_use]
            max_col_len = max(max_col_len, len(column))
            
        for i in range(max_col_len):
            formatters = []
            row = []
            for column, formatter in zip(column_contents, column_formatters):
                if i < len(column):
                    row += [column[i]]
                else: row += [""]
                formatters += [formatter[i]]

            formatter = "{left}" + "{separator}".join(formatters) + "{right}" + newline
            self.contents += formatter.format(*row, left=v, right=v, separator=v)
            
        self.contents += bl + bs.join([h*width for width in widths]) + br

        normal = starsmashertools.bintools.Style.get('text', 'normal')
        whitebg = starsmashertools.bintools.Style.get('backgrounds', 'white')
        graybg = starsmashertools.bintools.Style.get('backgrounds', 'light gray')
        contents = self.contents.split(newline)
        for i, c in enumerate(contents):
            bg = whitebg
            if i%2 == 0: bg = graybg
            if not c: continue

            c = c.replace(
                normal,
                normal + bg,
            ) + normal
            
            cstrip = c.strip()
            idx = c.index(cstrip[0])
            c = c[:idx] + bg + cstrip + normal
            contents[i] = c
        self.contents = newline.join(contents)
        
        super(Table, self).show(*args, **kwargs)
