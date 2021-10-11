import numpy as np
import matplotlib.pyplot as plt

epochs = 13500

train_loss = np.load('sigma1e-1_BN_trainloss.npy')
val_loss = np.load('sigma1e-1_BN_valloss.npy')
x = np.linspace(0,epochs,len(train_loss))
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-1$, batch normalization: on')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-1_BN_LOSS.png')
plt.clf()


train_loss = np.load('sigma1e-3_BN_trainloss.npy')
val_loss = np.load('sigma1e-3_BN_valloss.npy')
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-3$, batch normalization: on')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-3_BN_LOSS.png')
plt.clf()


train_loss = np.load('sigma1e-4_BN_trainloss.npy')
val_loss = np.load('sigma1e-4_BN_valloss.npy')
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-4$, batch normalization: on')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-4_BN_LOSS.png')
plt.clf()

#NO BN


train_loss = np.load('sigma1e-1_noBN_trainloss.npy')
val_loss = np.load('sigma1e-1_noBN_valloss.npy')
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-1$, batch normalization: off')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-1_noBN_LOSS.png')
plt.clf()


train_loss = np.load('sigma1e-3_noBN_trainloss.npy')
val_loss = np.load('sigma1e-3_noBN_valloss.npy')
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-3$, batch normalization: off')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-3_noBN_LOSS.png')
plt.clf()


train_loss = np.load('sigma1e-4_noBN_trainloss.npy')
val_loss = np.load('sigma1e-4_noBN_valloss.npy')
plt.plot(x,train_loss,x,val_loss)
plt.title('$\sigma = 1e-4$, batch normalization: off')
plt.xlabel('update step')
plt.ylabel('loss')
plt.legend(['training','validation'])
plt.savefig('sigma1e-4_noBN_LOSS.png')
plt.clf()

print(np.load('sigma1e-1_BN_testacc.npy'))
print(np.load('sigma1e-3_BN_testacc.npy'))
print(np.load('sigma1e-4_BN_testacc.npy'))

print(np.load('sigma1e-1_noBN_testacc.npy'))
print(np.load('sigma1e-3_noBN_testacc.npy'))
print(np.load('sigma1e-4_noBN_testacc.npy'))
