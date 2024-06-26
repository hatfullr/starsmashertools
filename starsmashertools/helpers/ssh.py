import subprocess
import os

#bash_command = "ssh -o PreferredAuthentications=publickey,keyboard-interactive -o KbdInteractiveAuthentication=yes -o ControlMaster=auto -o ControlPersist=300s -o BatchMode=yes -o ConnectTimeout=%d %s %s 2>&1"
bash_command = "ssh %s %s 2>&1"

def get_address(path):
    if ':' not in path: return None
    return split_address(path)[0]

def split_address(path):
    return path.split(":")

# This function determines of the given filename points to a local path,
# of if it points to a remote path on a server somewhere.
def isRemote(path):
    return get_address(path) is not None

def run(address, command, timeout=1, text=True):
    _command = bash_command % (address, command) #timeout, address, command)
    output = subprocess.check_output(
        _command,
        text=text,
        shell=True,
    )
    return output.strip()

def run_python(address, command, **kwargs):
    command = ('"' + "python3 -c " + r"\"" + "%s" + r"\"" + '"') % command
    return run(address, command, **kwargs)
