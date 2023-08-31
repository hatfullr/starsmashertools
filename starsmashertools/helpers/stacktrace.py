import traceback
import sys


# https://stackoverflow.com/a/16589622/4954083
# With some edits
def format_exc(exception=None, extra_message="", for_raise=False):
    if exception is not None: exc = type(exception)
    else: exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]    # remove call of full_stack, the printed exception
                         # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)

    if for_raise:
        stackstr = "\n".join(stackstr.split("\n")[:-2]) + "\n"
        sys.tracebacklimit = -1
        
    if extra_message:
        stackstr += "\n  "+extra_message+"\n"
        
    return stackstr
