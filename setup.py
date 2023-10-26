if __name__ == "__main__":
    from setuptools import setup, find_packages
    import inspect.metadata
    import os
    setup(
        name='starsmashertools',
        version=inspect.metadata.version('starsmashertools'),
        packages=find_packages(),
        scripts=[
            os.path.join('bin','starsmashertools'),
            os.path.join('bin','starsmasherplot'),
            os.path.join('bin','starsmasherdir'),
            os.path.join('bin','inspectpdc'),
            os.path.join('bin','outtotxt'),
            os.path.join('bin','editpdc'),
            os.path.join('bin','timing'),
        ],
        install_requires = [
            'numpy',
            'scipy',
            'setuptools>=61.0',
        ],
    )
