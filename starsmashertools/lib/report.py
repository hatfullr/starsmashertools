import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import typing
import collections
import starsmashertools.lib.simulation
import copy
import _io
import numpy as np
import ast
import inspect

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
            column_separator : str = '  ',
    ):
        self.column_separator = column_separator

        self._columns = []
        
        if columns is None:
            try: columns = self.preferences.get('columns')
            except: columns = []

        for obj in columns:
            self.add_column(*obj['args'], **obj['kwargs'])
        
        for simulation in simulations:
            self.add_row(simulation)

    def __str__(self):
        return '\n'.join(iter(self))
            
    def __iter__(self):
        self._current = 0
        return self
    def __next__(self):
        try:
            row = [str(column[self._current]) for column in self._columns]
        except IndexError as e:
            del self._current
            raise StopIteration from e
        self._current += 1
        return self.column_separator.join(row)


    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        if len(self._columns) != len(other._columns): return False
        if self.column_separator != other.column_separator: return False
        for c1, c2 in zip(self._columns, other._columns):
            if c1 != c2: return False
        return True
    def __neq__(self, other): return not (self == other)

    class Cell(object):
        def __init__(
                self,
                simulation : str | starsmashertools.lib.simulation.Simulation | type(None) = None,
                func : typing.Callable | type(None) = None,
                formatter : str = '{}',
                shorten : dict | type(None) = None,
        ):
            self.formatter = formatter
            self.shorten = shorten
            self._func = func
            self._value = None
            
            if simulation is not None:
                if isinstance(simulation, str):
                    simulation = starsmashertools.get_simulation(simulation)
            
            self.simulation = simulation
        
        @property
        def func(self): return self._func
        @func.setter
        def func(self, value):
            self._value = None
            self._func = value

        def __eq__(self, other):
            if not isinstance(other, self.__class__): return False
            if self.simulation != other.simulation: return False
            if self._value is None: self.update_value()
            if other._value is None: other.update_value()
            return self._value == other._value
        def __neq__(self, other): return not (self == other)

        def __str__(self):
            import starsmashertools.helpers.string
            
            if self._value is None: self.update_value()
            if self._value is None: return self.formatter.format('')
            
            string = self.formatter.format(self._value)
            if self.shorten is not None:
                string = starsmashertools.helpers.string.shorten(
                    string, *self.shorten['args'], **self.shorten['kwargs'],
                )
            return string

        # For pickling
        def __getstate__(self):
            # Check if the function is a lambda function
            if self._value is None: self.update_value()
            state = self.__dict__.copy()
            state['_func'] = None
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)

        def update_value(self):
            if None not in [self.simulation, self.func]:
                self._value = self.func(self.simulation)


    class Header(Cell, object):
        def __str__(self):
            if self._value is None: return self.formatter.format('')
            return self.formatter.format(self._value)

    class Column(object):
        def __init__(
                self,
                func : typing.Callable,
                formatter : str = '{}',
                header_formatter : str = '{}',
                header : str | type(None) = None,
                shorten : dict | type(None) = None,
        ):
            if header is None: header = func.__qualname__
            
            self._formatter = formatter
            self._shorten = shorten
            
            cell = Report.Header(
                formatter = header_formatter,
                func = func,
            )
            cell._value = header
            
            self._cells = [cell]

        def __len__(self): return len(self._cells)
        def __iter__(self): return iter(self._cells)

        def __eq__(self, other):
            if not isinstance(other, self.__class__): return False
            if len(self._cells) != len(other._cells): return False
            for c1, c2 in zip(self._cells, other._cells):
                if c1 != c2: return False
            return True

        def __getitem__(self, index):
            return self._cells[index]
        
        @property
        def formatter(self): return self._formatter
        @formatter.setter
        def formatter(self, value):
            for cell in self._cells[1:]: cell.formatter = value
            self._formatter = value
        @property
        def header_formatter(self): return self._cells[0].formatter
        @header_formatter.setter
        def header_formatter(self, value): self._cells[0].formatter = value
        @property
        def func(self): return self._cells[0].func
        @func.setter
        def func(self, value):
            for cell in self._cells: cell.func = value
        @property
        def header(self): return self._cells[0]._value
        @header.setter
        def header(self, value): self._cells[0]._value = value
        @property
        def shorten(self): return self._shorten
        @shorten.setter
        def shorten(self, value):
            self._shorten = value
            for cell in self._cells[1:]: cell.shorten = value
        
        def add(
                self,
                simulation : str | starsmashertools.lib.simulation.Simulation,
        ):
            if isinstance(simulation, str):
                simulation = starsmashertools.get_simulation(simulation)

            self._cells += [Report.Cell(
                simulation = simulation,
                func = self.func,
                formatter = self.formatter,
                shorten = self.shorten,
            )]

        def remove(
                self,
                simulation : int | str | starsmashertools.lib.simulation.Simulation,
        ):
            if isinstance(simulation, int):
                index = simulation
                if self._cells[index] == self._cells[0]: index += 1
            else:
                if isinstance(simulation, str):
                    simulation = starsmashertools.get_simulation(simulation)
                index = None
                for i, cell in enumerate(self._cells[1:]):
                    if cell.simulation != simulation: continue
                    index = i + 1
                    break
                else:
                    raise IndexError("No matching simulation found")
            del self._cells[index]

        def update(self, properties : dict):
            for key, val in properties.items():
                if hasattr(self, key): setattr(self, key, val)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def set_row(
            self,
            index : int,
            simulation : str | starsmashertools.lib.simulation.Simulation,
    ):
        if not self._columns:
            raise Exception("Rows can only be set after at least one column has been added")
        
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        
        for column in self._columns:
            for i, cell in enumerate(column):
                if i != index: continue
                cell.simulation = simulation
                break
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add_row(
            self,
            simulation : str | starsmashertools.lib.simulation.Simulation,
    ):
        if not self._columns:
            raise Exception("Rows can only be added to a Report after at least one column has been added")
        
        for column in self._columns:
            column.add(simulation)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def remove_row(
            self,
            simulation : int | str | starsmashertools.lib.simulation.Simulation,
    ):
        for column in self._columns:
            column.remove(simulation)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def set_column(self, header_string : str, **kwargs):
        if not self._columns:
            raise Exception("Columns can only be set after at least one column has been added")
        
        for column in self._columns:
            if header_string != column.header: continue
            kwargs['header'] = kwargs.get('header', column.header)
            column.update(kwargs)
            break
        else:
            raise Exception("Failed to find column")
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add_column(
            self,
            func : typing.Callable,
            header : str | type(None) = None,
            formatter : str = '{}',
            header_formatter : str = '{}',
            shorten : dict | type(None) = None,
    ):
        column = Report.Column(
            func = func,
            header = header,
            formatter = formatter,
            header_formatter = header_formatter,
            shorten = shorten,
        )
        if self._columns:
            for cell in self._columns[-1]:
                column.add(cell.simulation)
        self._columns += [column]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def remove_column(
            self,
            func : int | typing.Callable,
    ):
        if isinstance(func, int): del self._columns[func]
        else:
            index = None
            for i, column in enumerate(self._columns):
                if column.func != func: continue
                index = i
                break
            else: raise IndexError("No column found with function '%s'" % func)
            del self._columns[index]
    
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            filename : str,
    ):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.pickler

        if filename.endswith('.txt'):
            with starsmashertools.helpers.file.open(filename, 'w') as f:
                f.write(str(self))
        else:
            if not filename.endswith('.report'): filename += '.report'
            obj = starsmashertools.helpers.pickler.pickle_object(self)
            with starsmashertools.helpers.file.open(filename, 'wb') as f:
                f.write(obj)

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(filename : str):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.pickler
        
        try:
            with starsmashertools.helpers.file.open(filename, 'rb') as f:
                obj = f.read()
            return starsmashertools.helpers.pickler.unpickle_object(obj)
        except Exception as e:
            if not filename.endswith('.report'):
                raise Exception("Report files can only be loaded if they were written with the '.report' file extension") from e
            raise
