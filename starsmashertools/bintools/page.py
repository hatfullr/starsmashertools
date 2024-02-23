# A Page is an object which displays information in the terminal and is able to
# navigate to other Page objects.
import starsmashertools.bintools
import starsmashertools.bintools.inputmanager
import functools
import copy
import os
import inspect

newline = starsmashertools.bintools.Style.get('characters', 'newline')

class Page(object):
    def __init__(
            self,
            cli : "starsmashertools.bintools.cli.CLI",
            inputtypes,
            contents : str,
            header : str | type(None) = None,
            footer : str | type(None) = None,
            identifier : str | type(None) = None,
            inputmanager : starsmashertools.bintools.inputmanager.InputManager | type(None) = None,
            indent : int = 1,
            simulation_controls : bool = False,
            back : "Page | type(None)" = None,
            _quit : bool = True,
            callback = None,
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
        self.simulation_controls = simulation_controls
        self.callback = callback

        self.input = None

        self.events = {}

        self.connections = {}

        self._back = None
        self._back_asprompt = False
        if back is not None: self.add_back(back, asprompt=True)

    def prompt(self, refresh = True, **kwargs):
        if not kwargs: kwargs = self._prompt_kwargs
        if self.inputmanager is None:
            raise Exception("No InputManager was given for this page")
        self.input = self.inputmanager.get(self.inputtypes, refresh = refresh, **kwargs)
        return self.input

    def process_input(self, _input):
        if _input in self.triggers.keys():
            self.triggers[_input]()
            return True
        if self.simulation_controls and _input in ['s', 'simulations']:
            self.show_simulation_controls()
            return True
        if self._back is not None and _input in ['b', 'back']:
            self.back()
            return True
        if self._quit and _input in ['q', 'quit']:
            quit()
        return False

    def get_header(self):
        return copy.deepcopy(self.header)

    def get_footer(self):
        return copy.deepcopy(self.footer)

    def get_content(self):
        self.fire_events('get_content')
        
        if callable(self.contents): content = self.contents()
        else: content = copy.copy(self.contents)
        
        content = newline.join([" "*self.indent + c for c in content.split(newline)])
        
        header = self.get_header()
        footer = self.get_footer()
        if header is None: header = starsmashertools.bintools.Style.get('formatting', 'header')
        if footer is None: footer = starsmashertools.bintools.Style.get('formatting', 'footer')
        
        if header: content = header + content
        if footer: content += footer

        return newline.join([" "+c for c in content.split(newline)])

    def get_content_options(
            self,
            back : bool | type(None) = None,
            _quit : bool | type(None) = None,
            simulation_controls : bool | type(None) = None,
    ):
        content = []
        if simulation_controls is None:
            if self.simulation_controls:
                content += ['s) simulations']
        elif self.simulation_controls:
            content += ['s) simulations']
        
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
        return newline.join(content)
        

    # Keywords go to the input manager
    # Override this method in children, but also call this method from children
    # None in keywords means use the page's default
    def show(self, **kwargs):
        self.cli.page = self
        
        self.show_content()
        self.show_content_options(**kwargs)
        
        if not self.show_prompt(**kwargs): return
        self._on_no_connection()

    def show_content(self):
        content = self.get_content()
        if content:
            self.cli.write(content, flush=True)
            self.on_event_end('get_content')

    def show_content_options(self, **kwargs):
        options = self.get_content_options(**kwargs)
        if options:
            self.cli.write(options, flush=True)
            self.on_event_end('get_content_options')

    def show_prompt(self, refresh = True, **kwargs):
        if self.inputmanager is not None and self.inputtypes:
            self._prompt_kwargs = kwargs
            self.prompt(refresh = refresh)
            if self.process_input(self.input):
                if self.callback is not None: self.callback(self.input)
                return False
            if self.callback is not None: self.callback(self.input)
        
        if not self._back_asprompt and self._back is not None:
            self.back()
            return False
        return True

    def _on_no_connection(self):
        self.cli.reset()
            

    # Connecting two pages informs the CLI how to navigate the pages. The
    # 'triggers' argument should be an iterable whose elements are possible user
    # inputs.
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def connect(self, other : "Page | str", triggers):
        if not hasattr(triggers, '__iter__') or isinstance(triggers, str):
            raise TypeError("Argument 'triggers' must be a non-str iterable")
        if not isinstance(other, Page): # It's an identifier
            other = self.cli.pages[other]
        for trigger in triggers:
            if trigger in self.triggers.keys():
                raise KeyError("Trigger already exists: '%s'" % str(trigger))
            self.triggers[trigger] = lambda: self.cli.navigate(other.identifier)
        self.connections[other] = triggers

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def disconnect(self, other : "Page | str"):
        if not isinstance(other, Page): # It's an identifier
            other = self.cli.pages[other]
        if not self.connected(other):
            raise ValueError("Page '%s' is not connected to page '%s'" % (str(other), str(self)))
        for trigger in self.connections[other]:
            self.triggers.pop(trigger)
        self.connections.pop(other)

    def connected(self, other : "Page | str"):
        if not isinstance(other, Page): # It's an identifier
            other = self.cli.pages[other]
        return other in self.connections.keys()

    def back(self):
        if self._back is not None:
            self.on_back()
            self.cli.navigate(self._back.identifier)
        else:
            raise Exception("Tried to go back but no back page was set")

    # Override this
    def on_back(self, *args, **kwargs):
        pass

    def add_back(self, page, asprompt=True):
        self._back = page
        self._back_asprompt = asprompt
        if self._back_asprompt:
            if str not in self.inputtypes:
                self.inputtypes += [str]

    def show_simulation_controls(self):
        page = SimulationControlsPage(self.cli)
        self.cli.navigate(page)

    def add_listener(self, name : str, function, args=(), kwargs={}):
        """
        name = the name of a function belonging to this class
        """
        if name not in self.events.keys(): self.events[name] = []
        self.events[name] += [PageEvent(function, args, kwargs)]
    
    def remove_listener(self, name : str, function):
        new_list = []
        for i, event in enumerate(self.events[name]):
            if event.function is not function: new_list += [event]
        self.events[name] = new_list

    def fire_events(self, name):
        if name in self.events.keys():
            for event in self.events[name]: event()

    def on_event_end(self, name):
        if name in self.events.keys():
            for event in self.events[name]: event.reset()
        

class PageEvent(object):
    """
    This is for processing events triggered by pages
    """
    def __init__(self, function, args, kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.called = False

    def reset(self):
        self.called = False

    def __call__(self):
        if not self.called:
            ret = self.function(*self.args, **self.kwargs)
            self.called = True
            return ret


    

class List(Page, object):
    def __init__(self, cli, inputtypes, items=[], bullet="%d)", separator=' ', **kwargs):
        super(List, self).__init__(cli, inputtypes, "", **kwargs)
        self.items = []
        self.bullet = bullet
        self.separator = separator
        for item in items: self.add(item)

    def get_content(self, *args, **kwargs):
        self.fire_events('get_content')
        self.contents = []
        i = 0
        for item in self.items:
            text = item.get_text()
            if text:
                if item.bullet is None:
                    if self.bullet is not None:
                        item.bullet = self.bullet % i
                if item.separator is None:
                    if self.separator is not None:
                        item.separator = self.separator
                string = ''
                if item.bullet: string += item.bullet
                if item.separator: string += item.separator
                string += text
                self.contents += [string]
                i += 1
        self.contents = newline.join(self.contents)
        return super(List, self).get_content(*args, **kwargs)
        
    def add(self, string, **kwargs):
        self.items += [List.Item(string, **kwargs)]

    def _on_no_connection(self):
        error = starsmashertools.bintools.inputmanager.InputManager.InvalidInputError("Invalid list selection '%s'" % str(self.input))
        self.inputmanager.reset()
        starsmashertools.bintools.print_error(error=error)
        result = self.process_input(self.prompt())
        if result: return
        else: self._on_no_connection()


    class Item(object):
        def __init__(self, text, bullet=None, separator=None):
            self.text = text
            self.bullet = bullet
            self.separator = separator

        def get_text(self):
            if callable(self.text): return self.text()
            return self.text

        
class ConfirmationPage(List, object):
    """
    Ask the user for confirmation of something (yes/no question)
    """
    def __init__(
            self, 
            cli : "starsmashertools.bintools.cli.CLI",
            yes : str = 'yes',
            no : str = 'no',
            header : str = "Are you sure?"+newline,
            **kwargs
    ):
        self._yes = yes
        self._no = no
        super(ConfirmationPage, self).__init__(
            cli,
            [int, str],
            items = [self._no, self._yes],
            header=header,
            **kwargs
        )
        
    def connect_yes(self, page):
        index = None
        for i, item in enumerate(self.items):
            if item.get_text() == self._yes:
                index = i
                break
        self.connect(page, [index, self._yes])
    def connect_no(self, page):
        index = None
        for i, item in enumerate(self.items):
            if item.get_text() == self._no:
                index = i
                break
        self.connect(page, [index, self._no])
    def _on_no_connection(self):
        self.back()

        

class SimulationControlsPage(List, object):
    def __init__(self, cli):
        # This is effectively the main page

        if 'SimulationControlsPage' in cli.pages.keys():
            cli.remove_page('SimulationControlsPage')
        
        super(SimulationControlsPage, self).__init__(
            cli,
            [str],
            bullet = None,
            back = cli.page,
            _quit = True,
            simulation_controls = False,
            identifier='SimulationControlsPage',
        )
        self.remove_page = None
        
        cli._add_page(self)
        
        add_page = self.cli._add_page(SimulationControlsPage.AddSimulationPage(
            self.cli,
            [str],
            newline+"Enter a simulation directory to add"+newline+("Current working directory = '%s'" %  os.getcwd()),
            identifier='SimulationControlsPage.AddSimulationPage',
            back=self,
            simulation_controls = False,
        ))
        self.connect(add_page, [0, 'a', 'add'])

        if len(self.cli.simulations) >= 1:
            self.create_remove_page()
        

    def create_remove_page(self):
        if 'SimulationControlsPage.remove_page' in self.cli.pages.keys():
            self.cli.remove_page('SimulationControlsPage.remove_page')
        self.remove_page = self.cli.add_list(
            [int, str],
            items = [s.directory for s in self.cli.simulations],
            bullet = '%5d)',
            header= newline + "Choose a simulation:"+newline,
            back = self,
            _quit = True,
            simulation_controls = False,
            identifier = 'SimulationControlsPage.remove_page',
        )
        self.connect(self.remove_page, [1, 'r', 'remove'])

        for i, simulation in enumerate(self.cli.simulations):
            page = self.cli.add_page(
                [],
                lambda simulation=simulation: self.remove_simulation(simulation),
                _quit=False,
                simulation_controls = False,
            )
            page.add_back(self.remove_page if len(self.cli.simulations) > 1 else self, asprompt=False)
            self.remove_page.connect(page, [i, simulation.directory])

    def delete_remove_page(self):
        if self.remove_page is not None:
            self.cli.remove_page(self.remove_page)
            self.remove_page = None
        

    def show(self, *args, **kwargs):
        self.delete_remove_page()
        
        self.items = []
        self.add('%5s) add' % 'a')
        if self.cli.simulations:
            self.add('%5s) remove' % 'r')

            self.create_remove_page()
            
        self.header = newline + "Current simulations:" + newline + newline.join(["   %s" % s.directory for s in self.cli.simulations]) + newline + newline + 'Choose an option:' + newline
        
        super(SimulationControlsPage, self).show(*args, **kwargs)

    def remove_simulation(self, simulation : starsmashertools.lib.simulation.Simulation):
        self.cli.remove_simulation(simulation)

        self.delete_remove_page()
        self.create_remove_page()
        
        return "Removed %s" % simulation.directory
        
    def on_back(self, *args, **kwargs):
        self.cli.remove_page(self)
    
    class AddSimulationPage(Page, object):
        def process_input(self, _input):
            if self._back is not None and _input in ['b', 'back']:
                self.cli.navigate(self._back.identifier)
            if self._quit and _input in ['q', 'quit']:
                quit()
                
            try:
                simulation = starsmashertools.get_simulation(_input)
                self.cli.simulations += [simulation]
                self.cli.write("Added simulation '%s'" % simulation.directory)
                self.back()
                return True
            except (starsmashertools.lib.simulation.Simulation.InvalidDirectoryError, FileNotFoundError) as e:
                starsmashertools.bintools.print_error(error=e)
                return False

        def _on_no_connection(self):
            error = starsmashertools.bintools.inputmanager.InputManager.InvalidInputError("Invalid StarSmasher simulation directory")
            starsmashertools.bintools.print_error(error=error)
            result = self.process_input(self.prompt())
            if result: return
            else: self._on_no_connection()
        


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










class MultiPage(Page, object):
    """
    Splits the content of a single page into multiple pages with navigation.
    This is useful for a Page which contains too much information to fit on the
    screen.
    """
    def __init__(self, *args, **kwargs):
        super(MultiPage, self).__init__(*args, **kwargs)
        self.contents_per_page = []
        self.pagenumber = 0
        self._forward = True
        self._listen_to_enter = 'forward'
        self._past_initial_input = False
        self._searching = False
        self._search_matches = []
        self._search_match_index = -1

    @property
    def width(self):
        height, width = self.cli.get_height_and_width()
        return width - 3

    def reverse(self):
        self._forward = False
        self.pagenumber = max(self.pagenumber - 1, 0)
        if self.pagenumber == 0:
            self._forward = True
            if self._past_initial_input:
                self._listen_to_enter = None
        
    def advance(self):
        self._forward = True
        self.pagenumber = min(self.pagenumber + 1, len(self.contents_per_page) - 1)
        if self.pagenumber == len(self.contents_per_page) - 1:
            self._forward = False
            if self._past_initial_input:
                self._listen_to_enter = None

    def goto(self, number):
        if number == 0:
            self._forward = True
            self._listen_to_enter = 'forward'
        elif number == len(self.contents_per_page) - 1:
            self._forward = False
            self._listen_to_enter = 'backward'
        
        self.pagenumber = number

    def previous_match(self):
        self._search_match_index = max(self._search_match_index - 1, 0)
        self.show_search_match()
        self._listen_to_enter = 'search backward'

    def next_match(self):
        self._search_match_index = min(
            self._search_match_index + 1,
            len(self._search_matches) - 1,
        )
        self.show_search_match()
        self._listen_to_enter = 'search forward'

    def prompt(self, **kwargs):
        kwargs['error_on_empty'] = False
        return super(MultiPage, self).prompt(**kwargs)
    
    def process_input(self, _input):
        import starsmashertools.bintools
        import starsmashertools.bintools.inputmanager

        if _input in self.triggers.keys():
            self.triggers[_input]()
            return True
        if _input == '.':
            self._listen_to_enter = 'forward'
            self._forward = True
            self.advance()
            return False
        if _input == ',':
            self._listen_to_enter = 'backward'
            self._forward = False
            self.reverse()
            return False
        if _input is None or (isinstance(_input, str) and _input.strip() == ''):
            if self._listen_to_enter == 'forward' and self._forward:
                self.advance()
            elif self._listen_to_enter == 'backward' and not self._forward:
                self.reverse()
            elif self._listen_to_enter == 'search forward' and self._searching:
                self.next_match()
            elif self._listen_to_enter == 'search backward' and self._searching:
                self.previous_match()
            return False
        if _input == 'a':
            self.goto(0)
            return False
        if _input == 'e':
            self.goto(len(self.contents_per_page) - 1)
            return False
        
        if self._searching:
            if self._search_matches:
                if _input == '>':
                    self.next_match()
                    return False
                if _input == '<':
                    self.previous_match()
                    return False

            if _input in ['', None]:
                return False
            
            self._search_matches = self.search(_input)
            if self._search_matches:
                self.next_match()
                #self._search_match_index = 0
                #self.show_search_match()
                return False
            else:
                raise starsmashertools.bintools.inputmanager.InputManager.InvalidInputError("phrase not found: %s" % repr(_input))
            
            return False
        
        
        if _input == 'S':
            self._searching = True
            return False
        if self.simulation_controls and _input in ['s', 'simulations']:
            self.show_simulation_controls()
            return True
        if self._back is not None and _input in ['b', 'back']:
            self.back()
            return True
        if self._quit and _input in ['q', 'quit']:
            quit()
        return False
    
    def show(self, **kwargs):
        import starsmashertools.bintools.inputmanager
        import starsmashertools.bintools
        self.cli.page = self
        
        if not self.contents_per_page: self.split_content_to_pages(**kwargs)

        self.cli.clear()
        self.show_content()
        self.show_content_options(**kwargs)

        while True:
            try:
                prompt_answer = self.show_prompt(**kwargs)
            except (KeyboardInterrupt, starsmashertools.bintools.inputmanager.InputManager.InvalidInputError) as e:
                if isinstance(e, KeyboardInterrupt):
                    if self._searching:
                        self.reset_search()
                        self.cli.clear()
                        self.show_content()
                        self.show_content_options(**kwargs)
                        continue
                    else: raise
                else:
                    starsmashertools.bintools.print_error(error = e, halt = False)
                    continue
            
            if prompt_answer:
                self._past_initial_input = True
                self.cli.clear()
                self.show_content()
                self.show_content_options(**kwargs)
            else: break
            
        self._on_no_connection()

    def reset_search(self):
        self._search_match_index = -1
        self._searching = False
        self._search_matches = []

        # Redraw to remove highlighting
        self.cli.clear()
        self.show_content()
        self.show_content_options()
        

    def show_content(self, *args, **kwargs):
        if self.contents_per_page:
            if self._searching: content = self.get_highlighted()
            else: content = self.contents_per_page[self.pagenumber]
            content = self.get_page_header() + content + self.get_page_footer()
            self.cli.write(content, flush=True)

    def get_page_header(self):
        normal = starsmashertools.bintools.Style.get('text', 'normal')
        return normal + newline.join([
            '{pagenumber:<4d} / {total:<4d}  ,) previous {f1:7s}  .) next {f2:7s}  a) start  e) end  S) search',
            '┌' + '─' * self.width + '┐',
        ]).format(
            pagenumber = self.pagenumber + 1,
            f1 = '[enter]' if (self._listen_to_enter == 'forward' and not self._forward) else '',
            f2 = '[enter]' if (self._listen_to_enter == 'backward' and self._forward) else '',
            total = len(self.contents_per_page),
        )
    
    def get_page_footer(self):
        return '└' + '─' * self.width + '┘'

    def get_content_options(self, *args, **kwargs):
        if self._searching:
            exit_message = "Exit: Ctrl+C"
            if not self._search_matches:
                match_string = "Match -/-"
            else:
                total = len(self._search_matches)
                fmt = "Match {{current:{width:d}d}} / {{total:{width:d}d}}"
                fmt = fmt.format(width = len(str(total)))

                match_string = fmt.format(
                    current = self._search_match_index + 1,
                    total = total,
                )

            string = [
                '>) next match {:7s}'.format(
                    '[enter]' if self._listen_to_enter == 'search forward' else '',
                ),
                '<) previous match {:7s}'.format(
                    '[enter]' if self._listen_to_enter == 'search backward' else '',
                ),
            ]
            
            string[0] += ' ' * (self.width + 1 - len(string[0]) - len(match_string))
            string[0] += match_string
            string[1] += ' ' * (self.width + 1 - len(string[1]) - len(exit_message))
            string[1] += exit_message
            
            return newline.join(string)
        else:
            return super(MultiPage, self).get_content_options(*args, **kwargs)

    def split_content_to_pages(self, **kwargs):
        import textwrap
        import itertools
        
        height, width = self.cli.get_height_and_width()
        
        # Obtain all the content
        content = self.get_content()
        self.on_event_end('get_content')
        content_options = self.get_content_options(**kwargs)
        self.on_event_end('get_content_options')
        
        allowed_height = height - len(content_options.split(newline)) - 5
        allowed_width = self.width

        wrapper = textwrap.TextWrapper(width = allowed_width)
        content = [wrapper.wrap(i) for i in content.split(newline) if i != '']
        content = list(itertools.chain.from_iterable(content))
        
        for i, line in enumerate(content):
            if i == 0:
                self.contents_per_page = [[]]
            if i > 0 and i%allowed_height == 0:
                self.contents_per_page += [[]]
            missing_width = allowed_width - len(line)
            self.contents_per_page[-1] += ['│' + line + ' '*missing_width + '│']

        # Fill out each page to the maximum height
        empty_line = '│' + ' '*allowed_width + '│'
        for i, page in enumerate(self.contents_per_page):
            if len(page) < allowed_height:
                self.contents_per_page[i] = page + [empty_line]*(allowed_height - len(page))
            
        for i in range(len(self.contents_per_page)):
            self.contents_per_page[i] = newline.join(self.contents_per_page[i])


    def search(self, phrase):
        """
        Scan all contents to find the given all instances of the given phrase.
        """
        import re
        import starsmashertools.bintools
        import starsmashertools.bintools.inputmanager
        
        search_matches = []
        
        use = re.compile(phrase, flags = re.MULTILINE)

        for pagenumber, content in enumerate(self.contents_per_page):
            search_matches += [{
                'page number' : pagenumber,
                'start' : m.start(),
                'end' : m.end(),
                'page content' : content,
                'phrase' : phrase,
            } for m in use.finditer(content)]

        return search_matches


    def get_highlighted(self):
        if self._search_match_index == -1:
            return self.contents_per_page[self.pagenumber]
        
        style = starsmashertools.bintools.Style.get('text', 'bold')
        normal = starsmashertools.bintools.Style.get('text', 'normal')

        match = self._search_matches[self._search_match_index]
        pagenumber = match['page number']
        content = self.contents_per_page[pagenumber]

        i0, i1 = match['start'], match['end']
        content = normal + content[:i0] + style + content[i0:i1] + normal + content[i1:]
        return content
    

    def show_search_match(self):
        match = self._search_matches[self._search_match_index]
        if self.pagenumber != match['page number']:
            self.goto(match['page number'])
        
        self.cli.clear()
        self.show_content()
        self.show_content_options()

