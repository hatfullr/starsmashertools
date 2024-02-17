import pickle
import base64

def pickle_object(obj):
    return base64.b64encode(pickle.dumps(obj))

def unpickle_object(obj):
    return pickle.loads(base64.b64decode(obj))

    
