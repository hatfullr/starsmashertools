#import os
import subprocess

# This actually produces an error when something goes wrong
subprocess.check_output(
    'python3 -m unittest',
    shell=True,
    text=True,
)


#os.system("python3 -m unittest")

