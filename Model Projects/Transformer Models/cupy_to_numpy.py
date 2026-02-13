import pickle
import cupy as cp
import numpy as np

model_to_convert = "Models/tinychat_v2_100000.pkl"
output_model_path = "Models/numpy_tinychat_v2_100000.pkl"

def accuracy(prediction, label):
    num_correct = 0
    for i in range(len(prediction)):
        num_correct += np.argmax(prediction[i]) == np.argmax(label[i])

    return num_correct / len(prediction)

def cupy_to_numpy(obj):
    if isinstance(obj, cp.ndarray):
        return cp.asnumpy(obj)
    elif isinstance(obj, dict):
        return {k: cupy_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [cupy_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(cupy_to_numpy(v) for v in obj)
    elif isinstance(obj, set):
        return {cupy_to_numpy(v) for v in obj}
    else:
        return obj

with open(model_to_convert, "rb") as f:
    data = pickle.load(f)

data_np = cupy_to_numpy(data)

with open(output_model_path, "wb") as f:
    pickle.dump(data_np, f, protocol=pickle.HIGHEST_PROTOCOL)