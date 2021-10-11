import numpy as np
import matplotlib.pyplot as plt

lambda_dic = np.load('lambda_results_8_3.npy').item()
x = [float(key) for key in lambda_dic.keys()]
y = [val for val in lambda_dic.values()]

min_x = np.max(y)
print(x[y.index(min_x)])
plt.semilogx(x,y,'o')
plt.xlabel('$\lambda$')
plt.ylabel('Validation accuracy')
plt.show()
