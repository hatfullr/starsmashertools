import json
import gzip
import numpy as np
import starsmashertools.helpers.path

# Add your own serialization methods here to convert to/from JSON format.
serialization_methods = {
    # These types don't need conversions: # https://docs.python.org/3/library/json.html#json.JSONEncoder
    dict        : {'name' : 'dict' , 'conversions' : [lambda obj: obj, lambda obj: obj]},
    list        : {'name' : 'list' , 'conversions' : [lambda obj: obj, lambda obj: obj]},
    tuple       : {'name' : 'tuple', 'conversions' : [lambda obj: obj, lambda obj: obj]},
    str         : {'name' : 'str'  , 'conversions' : [lambda obj: obj, lambda obj: obj]},
    int         : {'name' : 'int'  , 'conversions' : [lambda obj: obj, lambda obj: obj]},
    float       : {'name' : 'float', 'conversions' : [lambda obj: obj, lambda obj: obj]},
    bool        : {'name' : 'bool' , 'conversions' : [lambda obj: obj, lambda obj: obj]},
    type(None)  : {'name' : 'None' , 'conversions' : [lambda obj: obj, lambda obj: obj]},

    # User conversions
    np.ndarray : {'name' : 'np.ndarray', 'conversions' : [
        lambda obj: obj.tolist(),    # To JSON
        lambda obj: np.asarray(obj), # From JSON
    ]},
}


def save(filename, obj):
    if '.gz' in filename:
        with gzip.open(filename, 'w') as f:
            try:
                f.write(json.dumps(obj, indent=4, cls=Encoder).encode('utf-8'))
            except Exception as e:
                if starsmashertools.helpers.path.isfile(filename):
                    starsmashertools.helpers.path.remove(filename)
                message = ("\nFailed to write file '%s'") % filename
                if isinstance(e, json.decoder.JSONDecodeError):
                    e.args = [e.args[0] + message]
                elif isinstance(e, TypeError):
                    if 'Object of type' in str(e) and 'is not JSON serializable' in str(e):
                        message += "\nPlease add a serialization method to the serialization_method dict in jsonfile.py"
                    e = TypeError(str(e) + message)
                raise(e)
    else:
        with open(filename, 'w') as f:
            try:
                f.write(json.dumps(obj, indent=4, cls=Encoder))
            except Exception as e:
                if starsmashertools.helpers.path.isfile(filename):
                    starsmashertools.helpers.path.remove(filename)
                message = ("\nFailed to write file '%s'") % filename
                if isinstance(e, json.decoder.JSONDecodeError):
                    e.args = [e.args[0] + message]
                elif isinstance(e, TypeError):
                    if 'Object of type' in str(e) and 'is not JSON serializable' in str(e):
                        message += "\nPlease add a serialization method to the serialization_method dict in jsonfile.py"
                    e = TypeError(str(e) + message)
                raise(e)

def load(filename):
    if '.gz' in filename:
        with gzip.open(filename, 'r') as f:
            try:
                ret = json.loads(f.read().decode('utf-8'), cls=Decoder)
            except json.decoder.JSONDecodeError as e:
                message = ("\nFailed to load file '%s'") % filename
                e.args = [e.args[0] + message]
                raise(e)
    else:
        with open(filename, 'r') as f:
            try:
                ret = json.load(f, cls=Decoder)
            except json.decoder.JSONDecodeError as e:
                message = ("\nFailed to load file '%s'") % filename
                e.args = [e.args[0] + message]
                raise(e)
    return ret



# These do the magic of encoding the types defined in the serialization_methods dict

class Encoder(json.JSONEncoder):
    def default(self, obj):
        _types = serialization_methods.keys()
        if isinstance(obj, tuple(_types)):
            m = serialization_methods[type(obj)]
            method = m['conversions'][0]
            if method is None: ret = obj
            ret = method(obj)
            return {'starsmashertools conversion name' : m['name'], 'value' : ret}
        return super(Encoder, self).default(obj)
class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        kwargs['object_hook'] = self.object_hook
        super(Decoder, self).__init__(*args, **kwargs)
        
    def object_hook(self, obj):
        if isinstance(obj, dict):
            name = obj.get('starsmashertools conversion name', None)
            if name is None: return obj
            for _type, vals in serialization_methods.items():
                if vals['name'] == name:
                    method = vals['conversions'][1]
                    if method is None: return obj
                    return method(obj['value'])
        return obj
