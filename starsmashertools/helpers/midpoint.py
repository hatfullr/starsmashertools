# Use the midpoint rule to search for an object in a list of objects which
# satisfies some given criteria. Set the criteria using the set_criteria method,
# where each argument 'low', 'equal', and 'high' are callable functions which
# take as input any of the objects you passed in when creating the Midpoint
# object.

import enum
import copy

class Midpoint(object):
    def __init__(self, objects, max_iter=10000):
        self.objects = objects
        self.max_iter = max_iter
        self.iteration = None

        self._criteria = None
        
    def set_criteria(self, low, equal, high):
        self._criteria = Midpoint.Criteria(low, equal, high)
    
    def get(self, favor='low'):
        if favor not in ['low', 'mid', 'high']:
            raise ValueError("favor must be one of 'low', 'mid', or 'high', not '%s'" % str(favor))
        
        if len(self.objects) == 1: return self.objects[0]
        
        low = 0
        high = len(self.objects) - 1

        results = [None] * len(self.objects)
        
        for self.iteration in range(self.max_iter):
            mid = int(0.5*(low + high))
            
            if mid == low or mid == high:
                if favor == 'low': return self.objects[low]
                elif favor == 'mid': return self.objects[mid]
                elif favor == 'high': return self.objects[high]
                raise RuntimeError("This should never be possible")
            
            if results[mid] is None:
                results[mid] = self._criteria.check(self.objects[mid])
            
            if results[mid] == Midpoint.Criteria.Result.LessThan:
                # Search higher in the list
                low = copy.copy(mid)
            elif results[mid] == Midpoint.Criteria.Result.Equal:
                return self.objects[mid]
            elif results[mid] == Midpoint.Criteria.Result.GreaterThan:
                # Search lower in the list
                high = copy.copy(mid)
            else:
                raise RuntimeError("This should never be possible")
        raise RuntimeError("Reached maximum iterations in '%s'" % str(self))

    class Criteria(object):
        class Result(enum.Enum):
            LessThan = 1
            Equal = 2
            GreaterThan = 3
        
        def __init__(self, low, equal, high):
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
            if self.low(obj): return Midpoint.Criteria.Result.LessThan
            elif self.equal(obj): return Midpoint.Criteria.Result.Equal
            elif self.high(obj): return Midpoint.Criteria.Result.GreaterThan
            raise RuntimeError("Object out of bounds: '%s'" % str(obj))
                
        class CriteriaNotCallableError(Exception, object): pass

