# Use the midpoint rule to search for an object in a list of objects which
# satisfies some given criteria. Set the criteria using the set_criteria method,
# where each argument 'low', 'equal', and 'high' are callable functions which
# take as input any of the objects you passed in when creating the Midpoint
# object.

import enum
import copy
import starsmashertools.helpers.argumentenforcer

class Midpoint(object):
    """
    This class provides an abstract way of implementing the midpoint method.
    A list of objects is specified at instantiation and a set of criteria are
    provided in `~.set_criteria`. Each criterion is a function which accepts one
    of the objects given in the list at instantiation and returns a boolean. The
    return values inform the method on how to take steps towards the midpoint
    value in the `~.get` function.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            objects : list | tuple,
    ):
        """
        The constructor for Midpoint class.

        Parameters
        ----------
        objects : list, tuple
            The objects which will be sent to the criteria functions during
            runtime.
        """
        self.objects = objects
        self.iteration = None

        self._criteria = None
        
    def set_criteria(self, low, equal, high):
        """
        Set the criterion functions to be evaluated by the `~.get` method. Each
        criterion is passed as input to the `~.Criteria` class constructor.

        Parameters
        ----------
        low : callable
            A function which accepts an object from the list of objects given in
            the constructor and returns `True` if the object represents a value
            lower than the value expected from the midpoint object and `False`
            otherwise.
        
        equal : callable
            A function which accepts an object from the list of objects given in
            the constructor and returns `True` if the object represents a value
            equal to the value expected from the midpoint object and `False`
            otherwise.

        high : callable
            A function which accepts an object from the list of objects given in
            the constructor and returns `True` if the object represents a value
            higher than the value expected from the midpoint object and `False`
            otherwise.

        See Also
        --------
        `~.Criteria`
        `~.get`
        """
        self._criteria = Midpoint.Criteria(low, equal, high)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get(
            self,
            favor : str = 'low',
            max_iter : int = 10000,
            return_index : bool = False,
    ):
        """
        Obtain the object at the midpoint. We work with 3 indices, `low`, `mid`,
        and `high`, where each index points to one of the objects in the list of
        objects specified in the constructor. The value of `low` is initially 0
        and `high` is `len(objects) - 1`. On every iteration, `mid` is evaluated
        as `int((low + high)/2)` and we check if `mid == low` or `mid == high`.
        If so, then we return the object specified by `favor`. Otherwise, we
        obtain the result of each criterion function for the objects in the list
        at `low`, `mid`, and `high`. If the "LessThan" criterion is `True`
        then we set `low = mid`, otherwise if the "Equal" criterion is `True`
        then we return `objects[mid]`. Otherwise, if the "GreaterThan" criterion
        is `True` then `high = mid` and we continue to the next iteration.

        Parameters
        ----------
        favor : str, default='low'
            In the case where the midpoint is equal to one of the bounding
            values (`low` or `high`) then `objects[low]` is returned if
            `favor == 'low'`, `objects[mid]` if `favor == 'mid'`, and
            `objects[high]` if `favor == 'high'`.

            Must be one of `'low'`, `'mid'`, or `'high'`.

        max_iter : int, default = 10000
            The maximum number of iterations.

        return_index : bool, default = False
            If `True`, the second value returned is the midpoint object's index
            in the array of objects.

        Returns
        -------
        object
            The member of the `objects` list given in the constructor that
            corresponds with the midpoint value as evaluated by the criteria.

        object, index

        See Also
        --------
        `~.__init__`
        `~.set_criteria`
        """
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'favor' : ['low', 'mid', 'high'],
        })

        def _return(index):
            if return_index: return self.objects[index], index
            return self.objects[index]
        
        if len(self.objects) == 1: return _return(0)
        
        low = 0
        high = len(self.objects) - 1

        results = [None] * len(self.objects)
        
        for self.iteration in range(max_iter):
            mid = int(0.5*(low + high))
            
            if mid == low or mid == high:
                if favor == 'low': return _return(low)
                elif favor == 'mid': return _return(mid)
                elif favor == 'high': return _return(high)
                raise RuntimeError("This should never be possible")
            
            if results[mid] is None:
                results[mid] = self._criteria.check(self.objects[mid])
            
            if results[mid] == Midpoint.Criteria.Result.LessThan:
                # Search higher in the list
                low = copy.copy(mid)
            elif results[mid] == Midpoint.Criteria.Result.Equal:
                return _return(mid)
            elif results[mid] == Midpoint.Criteria.Result.GreaterThan:
                # Search lower in the list
                high = copy.copy(mid)
            else:
                raise RuntimeError("This should never be possible")
        raise RuntimeError("Reached maximum iterations in '%s'" % str(self))

    class Criteria(object):
        """
        A class representing the criterion functions used in the `Midpoint`
        class.
        """
        class Result(enum.Enum):
            """
            An enumerator used to identify the result of the criterion functions
            used in the `Midpoint.Criteria` class.
            """
            LessThan = 1
            Equal = 2
            GreaterThan = 3
        
        def __init__(self, low, equal, high):
            """
            Criteria constructor.

            Parameters
            ----------
            low : callable
                A function which accepts an arbitrary object and returns a 
                value that is checked in the `~.check` method. This function
                should return a value that evaluates to `True` if the given
                object represents a case that is 'lower' than the midpoint
                value.
        
            equal : callable
                A function which accepts an arbitrary object and returns a 
                value that is checked in the `~.check` method. This function
                should return a value that evaluates to `True` if the given
                object represents a case that is 'equal' to the midpoint value.

            high : callable
                A function which accepts an arbitrary object and returns a 
                value that is checked in the `~.check` method. This function
                should return a value that evaluates to `True` if the given
                object represents a case that is 'higher' than the midpoint
                value.
            """
            if not callable(low):
                raise Midpoint.Criteria.CriteriaNotCallableError(str(low))
            if not callable(equal):
                raise Midpoint.Criteria.CriteriaNotCallableError(str(equal))
            if not callable(high):
                raise Midpoint.Criteria.CriteriaNotCallableError(str(high))
            self.low = low
            self.equal = equal
            self.high = high

        def check(self, obj):
            """
            Check the return values of the `low`, `equal`, and `high` functions
            given in the constructor, in that order. The first one to evaluate
            as `True` returns the return value.

            Parameters
            ----------
            obj : object
                An arbitrary object that is passed to the `low`, `equal`, and
                `high` functions.

            Returns
            -------
            Midpoint.Criteria.Result
                If the `low` function evaluates to `True`, `~.Result.LessThan`
                is returned. If the `mid` function evaluates to True`,
                `~.Result.Equal` is returned. If the `low` function evaluates to
                `True`, `~.Result.GreaterThan` is returned. Otherwise a
                `RuntimeError` is raised.

            See Also
            --------
            `~.Result`
            """
            if self.low(obj): return Midpoint.Criteria.Result.LessThan
            elif self.equal(obj): return Midpoint.Criteria.Result.Equal
            elif self.high(obj): return Midpoint.Criteria.Result.GreaterThan
            raise RuntimeError("Object out of bounds: '%s'" % str(obj))
                
        class CriteriaNotCallableError(Exception, object): pass

