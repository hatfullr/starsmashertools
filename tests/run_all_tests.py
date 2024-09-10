def run():
    import subprocess
    import basetest
    import sys

    # Extract the test simulation directories
    basetest.extract_archive(quiet = True, no_remove = True)
    
    exitcode = 0
    try:
        # This actually produces an error when something goes wrong
        p = subprocess.Popen(['python3', '-m', 'unittest'])
        p.wait()
    except:
        exitcode = 1

    # Re-compress the simulation directories asynchronously
    basetest.restore_archive()
    sys.exit(exitcode)

if __name__ == '__main__': run()

    

    





