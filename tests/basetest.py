import unittest

def isArchiveExtracted():
    import os
    import zipfile

    filename = 'archive.zip'
    if not os.path.isfile(filename): return True

    with zipfile.ZipFile(filename, mode='r') as zfile:
        for zinfo in zfile.infolist():
            directory = os.path.dirname(zinfo.filename)
            if not directory: continue

            if not os.path.exists(zinfo.filename):
                return False
    return True

    
class BaseTest(unittest.TestCase, object):
    @classmethod
    def setUpClass(cls):
        # Don't do this when using run_all_tests.py
        if cls.__module__ != '__main__': return
        if isArchiveExtracted(): return
        import subprocess
        p = subprocess.Popen(['./extract', '--quiet', '--no-remove'])
        p.wait()

    @classmethod
    def tearDownClass(cls):
        # Don't do this when using run_all_tests.py
        if cls.__module__ != '__main__': return
        if not isArchiveExtracted(): return
        import subprocess
        p = subprocess.Popen(['./restore', '--quiet'])
        p.wait()
        



    
