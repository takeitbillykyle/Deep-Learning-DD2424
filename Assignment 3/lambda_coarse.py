import numpy as np
import matplotlib.pyplot as plt

lambda_dic = np.load('lambda_acc.npy').item()
x = [float(key) for key in lambda_dic.keys()]
y = [val for val in lambda_dic.values()]

min_x = np.max(y)
plt.semilogx(x,y,'o')
plt.xlabel('$\lambda$')
plt.ylabel('Validation accuracy')
plt.show()
