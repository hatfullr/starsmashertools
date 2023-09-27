from setuptools import setup, find_packages
import os

setup(
    name='starsmashertools',
    version='0.0.1',
    packages=find_packages(),
    scripts=[
        os.path.join('bin','starsmasherdir'),
        os.path.join('bin','inspectpdc'),
        os.path.join('bin','outtotxt'),
        os.path.join('bin','editpdc'),
        os.path.join('bin','timing'),
    ],
)
