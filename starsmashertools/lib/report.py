import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import typing
import collections
import starsmashertools.lib.simulation
import starsmashertools.lib.units
import copy
import io

@starsmashertools.preferences.use
class Report(object):
    """
    Given Simulations, generate a report which shows their properties in
    provided text formatting.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            simulations : list | tuple,
            columns : list | tuple | type(None) = None,
    ):
        self.rows = []
        self.columns = collections.OrderedDict()

        if columns is None:
            try: columns = self.preferences.get('columns')
            except: columns = []
        
        for simulation in simulations:
            self.row(simulation)
        for obj in columns:
            self.column(*obj['args'], **obj['kwargs'])

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def write(
            self,
            stdout : io.TextIOWrapper | type(None) = None,
            column_separator : str = '  ',
            newline : str = '\n',
    ):
        import starsmashertools.helpers.string
        grid = []

        # Get all the rows first, so we can determine the column widths after
        widths = [0]*len(self.columns.keys())
        for simulation in self.rows:
            row = []
            for j, (_, column) in enumerate(self.columns.items()):
                result = column['func'](simulation)
                if isinstance(result, starsmashertools.lib.units.Unit):
                    result = result.auto()
                s = column['formatter'].format(result)
                shorten = column.get('shorten', None)
                if shorten is not None:
                    s = starsmashertools.helpers.string.shorten(
                        s, *shorten['args'], **shorten['kwargs'],
                    )
                        
                widths[j] = max(widths[j], len(s))
                row += [s]
            grid += [row]

        # Make the header, with the correct widths
        header_row = []
        for j, (header, column) in enumerate(self.columns.items()):
            fmt = '{:>' + str(widths[j]) + '}'
            header_row += [fmt.format(header)]
        grid = [header_row] + grid

        # Now combine the grid into a single string
        string = ""
        rows = []
        for row in grid:
            rows += [column_separator.join(row)]

        result = newline.join(rows)
        if stdout is None: return result
        else: stdout.write(result)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def row(
            self,
            simulation : str | starsmashertools.lib.simulation.Simulation,
    ):
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        self.rows += [simulation]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def column(
            self,
            func : typing.Callable,
            header : str | type(None) = None,
            formatter : str = '{}',
            shorten : dict | type(None) = None,
    ):
        if header is None: header = func.__qualname__
        
        self.columns[header] = {
            'func' : func,
            'formatter' : formatter,
            'shorten' : shorten,
        }
