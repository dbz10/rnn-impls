import re
import numpy as np

def load_data(path):
    lines = open(path, 'r').readlines()
    out = list(filter(None, [re.sub("\n", "", line) for line in lines]))
    return np.array(out)
