import unittest
import os
import trace
import starsmashertools
import shutil

class Ignore(trace._Ignore, object):
    def names(self, filename, modulename):
        if not filename.startswith(starsmashertools.SOURCE_DIRECTORY): return 1
        return super(Ignore, self).names(filename, modulename)

# Capture whatever trace.Trace tries to print in a file instead of to stdout
def trace_print(tracer, string, end = '\n', flush = False):
    tracer.f.write(string + end)
    if flush: tracer.f.flush()

class MyTrace(trace.Trace, object):
    r""" Define my own trace.Trace to ignore all modules that aren't 
    starsmashertools. """
    def __init__(self, path : str, *args, **kwargs):
        if os.path.exists(path): os.remove(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        self.f = open(path, 'w')
        super(MyTrace, self).__init__(*args, **kwargs)
        self.ignore = Ignore(
            kwargs.get('ignoremods', ()),
            kwargs.get('ignoredirs', ()),
        )

    

def isArchiveExtracted():
    import zipfile
    from importlib.util import spec_from_loader, module_from_spec
    from importlib.machinery import SourceFileLoader 

    spec = spec_from_loader('compress', SourceFileLoader('compress', 'compress'))
    compress = module_from_spec(spec)
    spec.loader.exec_module(compress)

    if not os.path.isfile(compress.zipname): return True

    with zipfile.ZipFile(compress.zipname, mode='r') as zfile:
        for zinfo in zfile.infolist():
            directory = os.path.dirname(zinfo.filename)
            if not directory: continue

            if not os.path.exists(zinfo.filename):
                return False
    return True

def extract_archive(quiet = False, no_remove = False):
    from importlib.util import spec_from_loader, module_from_spec
    from importlib.machinery import SourceFileLoader 

    spec = spec_from_loader('extract', SourceFileLoader('extract', 'extract'))
    extract = module_from_spec(spec)
    spec.loader.exec_module(extract)
    extract.extract(quiet = quiet, no_remove = no_remove)

def restore_archive():
    from importlib.util import spec_from_loader, module_from_spec
    from importlib.machinery import SourceFileLoader 

    spec = spec_from_loader('restore', SourceFileLoader('restore', 'restore'))
    restore = module_from_spec(spec)
    spec.loader.exec_module(restore)
    restore.restore()


class BaseTest(unittest.TestCase, object):
    # In the unit tests that inherit from this class, set trace to True to make
    # a file containing a full Python trace in the 'trace' directory.
    #trace = True # uncomment to trace all that haven't set trace = False
    
    @classmethod
    def setUpClass(cls):
        # Don't do this when using run_all_tests.py
        if cls.__module__ != '__main__': return
        
        if hasattr(cls, 'trace') and cls.trace:
            path = os.path.join('trace', cls.__name__ + '.trace')
            print("Creating trace: '%s'" % path)
            if os.path.exists(path): os.remove(path)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            
            cls.tracer = MyTrace(
                path,
                trace = True,
                timing = True,
            )
        
        if not isArchiveExtracted():
            extract_archive(quiet = True, no_remove = True)

    @classmethod
    def tearDownClass(cls):
        # Don't do this when using run_all_tests.py
        if cls.__module__ != '__main__': return
        if isArchiveExtracted(): restore_archive()
    
    def runTest(self, *args, **kwargs):
        if hasattr(self, 'tracer'):
            # intercept print functions from trace module
            trace.print = lambda *args, **kwargs: trace_print(
                self.tracer,
                *args,
                **kwargs
            )
            ret = self.tracer.runfunc(
                super(BaseTest, self).runTest,
                *args,
                **kwargs
            )
            # Reset print functions in trace module
            trace.print = print
            return ret
        return super(BaseTest, self).runTest(*args, **kwargs)

