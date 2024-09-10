# This may appear in a future version of starsmashertools...
"""
import unittest
import os
import basetest
import starsmashertools
import shutil
import tempfile

curdir = os.getcwd()
#testfile = os.path.join(curdir, 'testproj')

class TestProject(basetest.BaseTest):
    def setUp(self):
        # Move the existing project somewhere else temporarily
        shutil.move(
            starsmashertools.PROJECT_DIRECTORY,
            starsmashertools.PROJECT_DIRECTORY + '_orig',
        )
        os.mkdir(starsmashertools.PROJECT_DIRECTORY)
        
        self.testfile = os.path.join(curdir, 'testproj.ssproj')
        

    def tearDown(self):
        if os.path.exists(starsmashertools.PROJECT_DIRECTORY):
            shutil.rmtree(starsmashertools.PROJECT_DIRECTORY)
        
        shutil.move(
            starsmashertools.PROJECT_DIRECTORY + '_orig',
            starsmashertools.PROJECT_DIRECTORY,
        )

        if os.path.exists(self.testfile): os.remove(self.testfile)
    
    def test_save_project(self):
        files = [
            'file',
            os.path.join('lib', 'file'),
            os.path.join('mpl', 'image', 'file'),
        ]

        paths = [os.path.join(starsmashertools.PROJECT_DIRECTORY, f) for f in files]
        for path in paths:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            
            with open(path, 'w') as f:
                f.write('test')
        
        for path in paths:
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            
            starsmashertools.save_project(self.testfile)
            #with open(self.testfile, 'rb') as f:
            #    content = f.read()
            #self.assertGreater(len(content), 0)
            

if __name__ == '__main__':
    unittest.main(failfast=True)
"""
