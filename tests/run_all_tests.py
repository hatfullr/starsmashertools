#import os
import subprocess

# Extract the test simulation directories
p = subprocess.Popen(['./extract', '--quiet', '--no-remove'])
p.wait()

try:
    # This actually produces an error when something goes wrong
    p = subprocess.Popen(['python3', '-m', 'unittest'])
    p.wait()
except:
    # Re-compress the simulation directories asynchronously
    p = subprocess.Popen(['./restore'])
    p.wait()
    raise

# Re-compress the simulation directories asynchronously
p = subprocess.Popen(['./restore', '--quiet'])
p.wait()



