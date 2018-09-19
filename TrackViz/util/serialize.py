import os
import pickle
import random


def random6():
    return random.randint(100000, 999999)


def pickle_save(path, obj):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', path, e)
        return False
    return True


def pickle_load(path):
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return None

if __name__ == '__main__':
    pass