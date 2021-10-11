import matplotlib.pyplot as plt
import numpy as np

BOOK_LENGTH = 1107545
losses_dict = np.load('losses_nodes_adam.npy').item()

for key in losses_dict:
	y_s = np.array(losses_dict[key]).reshape(len(losses_dict[key]),1)
	x_s = np.arange(0,BOOK_LENGTH*2//25,10000)
	plt.plot(x_s,y_s,label='# nodes = ' +str(key))

plt.xlabel('iterations')
plt.ylabel('smooth loss')
plt.legend()
plt.show()
