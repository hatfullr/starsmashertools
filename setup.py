import setuptools
import os

SOURCE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

version_line_regex = "[vV][eE][rR][sS][iI][oO][nN]\\s*[=]\\s*[\"']\\d*[.]\\d*[.]\\d*[\"']"
version_regex = r"\d*[.]\d*[.]\d*"

def get_version():
    import re
    with open(os.path.join(SOURCE_DIRECTORY, "pyproject.toml"), 'r') as f:
        contents = f.read()
    matches = re.findall(version_line_regex, contents)
    if len(matches) == 0:
        raise Exception("Failed to find the current version number in the pyproject.toml file")
    elif len(matches) != 1:
        raise Exception("Found more than one version number in the pyproject.toml file")
    return re.findall(version_regex, matches[0])[0]

setuptools.setup(
    name='starsmashertools',
    author="Roger Hatfull",
    version=get_version(),
    url="https://github.com/hatfullr/starsmashertools",
    packages=setuptools.find_packages(),
    scripts=[
        os.path.join('bin','starsmashertools'),
        os.path.join('bin','starsmasherplot'),
        os.path.join('bin','starsmasherdir'),
        os.path.join('bin','outtotxt'),
        os.path.join('bin','timing'),
    ],
    install_requires = [
        'numpy>=1.21.5',
        'scipy',
        'setuptools>=61.0',
    ],
)
