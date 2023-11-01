import json
import gzip
import zipfile
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

    np.int8     : {'name' :  'np.int8', 'conversions' : [lambda obj: int(obj), lambda obj : np.int8(obj)]},
    np.int16    : {'name' : 'np.int16', 'conversions' : [lambda obj: int(obj), lambda obj : np.int16(obj)]},
    np.int32    : {'name' : 'np.int32', 'conversions' : [lambda obj: int(obj), lambda obj : np.int32(obj)]},
    np.int64    : {'name' : 'np.int64', 'conversions' : [lambda obj: int(obj), lambda obj : np.int64(obj)]},

    np.uint8     : {'name' :  'np.uint8', 'conversions' : [lambda obj: int(obj), lambda obj : np.uint8(obj)]},
    np.uint16    : {'name' : 'np.uint16', 'conversions' : [lambda obj: int(obj), lambda obj : np.uint16(obj)]},
    np.uint32    : {'name' : 'np.uint32', 'conversions' : [lambda obj: int(obj), lambda obj : np.uint32(obj)]},
    np.uint64    : {'name' : 'np.uint64', 'conversions' : [lambda obj: int(obj), lambda obj : np.uint64(obj)]},

    np.float16    : {'name' :  'np.float16', 'conversions' : [lambda obj: float(obj), lambda obj : np.float16(obj)]},
    np.float32    : {'name' :  'np.float32', 'conversions' : [lambda obj: float(obj), lambda obj : np.float32(obj)]},
    np.float64    : {'name' :  'np.float64', 'conversions' : [lambda obj: float(obj), lambda obj : np.float64(obj)]},
    np.float128   : {'name' : 'np.float128', 'conversions' : [lambda obj: float(obj), lambda obj : np.float128(obj)]},

    np.bool_    : {'name' :  'np.bool_', 'conversions' : [lambda obj: bool(obj), lambda obj : np.bool_(obj)]},
    np.bytes_   : {'name' :  'np.bytes_', 'conversions' : [lambda obj: bytes(obj), lambda obj : np.bytes_(obj)]},
    np.str_     : {'name' :  'np.str_', 'conversions' : [lambda obj: str(obj), lambda obj : np.str_(obj)]},

    # User conversions
    np.ndarray : {'name' : 'np.ndarray', 'conversions' : [
        lambda obj: obj.tolist(),    # To JSON
        lambda obj: np.asarray(obj), # From JSON
    ]},
}






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

def save_bytes(obj, encoder=Encoder):
    return json.dumps(obj, indent=4, cls=encoder).encode('utf-8')

def save(filename, obj, encoder=Encoder):
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'w')
        writemethod = lambda o: f.write(save_bytes(o, encoder=encoder))
    elif filename.endswith('.zip'):
        f = zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
        writemethod = lambda o: f.writestr(starsmashertools.helpers.path.basename(filename).replace(".zip",""), save_bytes(o, encoder=encoder))
    else:
        f = open(filename, 'w')
        writemethod = lambda o: f.write(json.dumps(o, indent=4, cls=encoder))
            
    try:
        writemethod(obj)
    except Exception as e:
        f.close()
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
    f.close()

def load(filename, decoder=Decoder):
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'r')
        readmethod = lambda: load_bytes(f.read(), decoder=decoder)

    elif filename.endswith('.zip'):
        f = zipfile.ZipFile(filename, 'r', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
        readmethod = lambda: load_bytes(f.read(starsmashertools.helpers.path.basename(filename).replace(".zip","")), decoder=decoder)
    else:
        f = open(filename, 'r')
        readmethod = lambda: json.load(f, cls=decoder)
    
    try:
        ret = readmethod()
    except json.decoder.JSONDecodeError as e:
        f.close()
        message = ("\nFailed to load file '%s'") % filename
        e.args = [e.args[0] + message]
        raise(e)
    f.close()
    return ret

def load_bytes(content, decoder=Decoder):
    return json.loads(content.decode('utf-8'), cls=decoder)
