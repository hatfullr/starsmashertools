import numpy as np

sctypes = np.sctypes

types = {
    'real' : [float, int] + sctypes['float'] + sctypes['int'] + sctypes['uint'],
    'float' : [float] + sctypes['float'],
    'int' : [int] + sctypes['int'] + sctypes['uint'],
}
