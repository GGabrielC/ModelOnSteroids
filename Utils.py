import time
import pickle
import os
from functools import wraps

def printOutput(function_to_decorate):
    @wraps(function_to_decorate)
    def wrapper(*args, **kwargs):
        out = function_to_decorate(*args, **kwargs)
        print()
        print(out) if type(out) not in [list, tuple] else [print(o) for o in out]
        print()
        return out
    return wrapper

def track(function_to_decorate):
    @wraps(function_to_decorate)
    def wrapper(*args, **kwargs):
        print("+ {0}()".format(function_to_decorate.__name__))
        t = time.time()
        out = function_to_decorate(*args, **kwargs)
        t2 = time.time()
        te = (t2 - t) / 60
        ts = "[{0:.1f}min]".format(te) if te >= 0.1 else ""
        print("- {0}() {1}".format(function_to_decorate.__name__, ts))
        return out
    return wrapper

def outputIsStorableOnDiskForFutureFunctionCalls(function_to_decorate):
    @wraps(function_to_decorate)
    def wrapper(*args, **kwargs):
        fileName = f"output\\{function_to_decorate.__name__}.pkl"
        fileExists = os.path.isfile(fileName)
        out = loadFromDisk(fileName) if fileExists else function_to_decorate(*args, **kwargs)
        if not fileExists:
            saveOnDisk(out, fileName)
        return out
    return wrapper

def toLinesOfStr(iterable):
    l=list(iterable)
    return ("{}\n"*len(l)).format(*l)

@track
def saveOnDisk(data, fileName):
    with open(fileName, 'wb') as file:
        pickle.dump(data, file)

@track
def loadFromDisk(fileName):
    with open(fileName, 'rb') as file:
        data = pickle.load(file)
    return data

