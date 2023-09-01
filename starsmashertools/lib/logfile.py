import starsmashertools.preferences as preferences
import starsmashertools.helpers.path
import starsmashertools.helpers.file
from glob import glob


def find(directory, pattern=None, throw_error=False):
    if pattern is None:
        pattern = preferences.get_default('LogFile', 'file pattern', throw_error=True)
    tosearch = starsmashertools.helpers.path.join(
        starsmashertools.helpers.path.realpath(directory),
        pattern,
    )

    matches = glob(tosearch)
    if matches: matches = sorted(matches)
    elif throw_error: raise FileNotFoundError("No log files matching pattern '%s' in directory '%s'" % (pattern, directory))

    return matches

    
class LogFile(object):
    def __init__(self, path):
        self.path = starsmashertools.helpers.path.realpath(path)
        self._contents = None

    @property
    def contents(self):
        if self._contents is None: self.read()
        return self._contents

    def read(self):
        self._contents = ""
        f = starsmashertools.helpers.file.open(self.path, 'r')
        for line in f:
            if 'output: end of iteration' in line: break
            self._contents += line
        f.close()

    def get(self, phrase, end="\n"):
        if phrase not in self.contents: raise LogFile.PhraseNotFoundError("Failed to find '%s' in '%s'" % (phrase, self.path))
        i0 = self.contents.index(phrase) + len(phrase)
        i1 = i0 + self.contents[i0:].index(end)
        return self.contents[i0:i1]



    class PhraseNotFoundError(Exception):
        pass
