import numpy as np
import matplotlib.pyplot as plt

def cyclic_rate(eta_min,eta_max,time,step):
	l=time//(2*step)
	if (time//step)%2==0:
		return eta_min+(time-2*l*step)*(eta_max-eta_min)/step
	else:
		return eta_max-(time-(2*l+1)*step)*(eta_max-eta_min)/step 
def cyclic_rate_sin(eta_min,eta_max,time,step):
	return eta_min+(eta_max-eta_min)*np.abs(np.sin(0.5*np.pi*time/step))

eta_min = 1
eta_max = 5
step = 4
a=np.load('noise_results.npy').item()
plt.plot(np.array(list(a.keys()),dtype=float),np.array(list(a.values()),dtype=float),'*')
plt.show()
y1 = []
y2 = []
for time in range(12):
	y1.append(cyclic_rate(eta_min,eta_max,time,step))
	y2.append(cyclic_rate_sin(eta_min,eta_max,time,step))
plt.plot(np.arange(12),y1,np.arange(12),y2,'--')
plt.show()