import joblib

def save_as_pickle(obj, path):
    joblib.dump(obj, path)

def load_pickle(path):
    return joblib.load(path)
