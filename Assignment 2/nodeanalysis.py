import numpy as np
import matplotlib.pyplot as plt

data = np.load('parameter_study_300nodes.npy')
data = data.item()
keys = np.array(list(data.keys()),dtype=float)
values = list(data.values())
print(keys[values.index(np.max(values))])
print(max(values))