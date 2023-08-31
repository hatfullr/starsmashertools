from setuptools import setup, find_packages
import os

setup(
    name='starsmashertools',
    version='0.0.1',
    packages=find_packages(),
    scripts=[
        os.path.join('bin','inspectpdc'),
        os.path.join('bin','outtotxt'),
    ],
)
