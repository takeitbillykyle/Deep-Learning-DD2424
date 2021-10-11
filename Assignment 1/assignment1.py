import numpy as np
import matplotlib.pyplot as plt


datapath = 'C:\\Users\\arvid\\Documents\\KTH\\Masterkurser\\Deep Learning\\Assignments\\Assignment 1\\'

np.random.seed(137)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encoding(labels):
	t = np.concatenate((np.arange(len(labels)).reshape(len(labels),1),np.array(labels).reshape(len(labels),1)),axis=1)
	one_hot = np.zeros((len(labels),10))
	one_hot[t[:,0],t[:,1]] = 1
	return one_hot.T

def load_batch(file):
	data_as_dict = unpickle(file)
	img_data = data_as_dict[b'data']
	labels = data_as_dict[b'labels']
	labels_encoded = one_hot_encoding(labels)
	return img_data, labels, labels_encoded

def preprocess(tr_data,val_data,test_data):
	tr_mean = np.mean(tr_data,axis=0)
	tr_std = np.std(tr_data,axis=0)

	tr_data = (tr_data - tr_mean)/tr_std
	val_data = (val_data - tr_mean)/tr_std
	test_data = (test_data - tr_mean)/tr_std

	return tr_data.T, val_data.T, test_data.T

def initializeWeights():
	W = np.random.normal(0,0.01,size=3072*10).reshape(10,3072)
	b = np.random.normal(0,0.01,size=10).reshape(10,1)

	return W,b

def softMax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)

def evaluateClassifier(X,W,b):
	s = W@X+b
	return softMax(s)

def computeCost(X,Y,W,b,Lambda):
	p = evaluateClassifier(X,W,b)
	J = Lambda*np.sum(W**2)-1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@p)))
	return J

def computeAccuracy(X,y,W,b):
	p = evaluateClassifier(X,W,b)
	kstar = np.argmax(p,axis=0)
	acc = np.sum(kstar==y)/len(y)
	return acc

def computeGradient_SVM(X,Y,W,Lambda):
	n = np.shape(X)[1]
	s = evaluateClassifier(X,W,b)
	scores_true=np.sum(s*Y,axis=0)
	margins = s-scores_true+1
	ind = margins>0
	g = ind-Y
	sumOfG=np.sum(g,axis=0)
	g=np.where(Y!=1,g,-sumOfG)
	dJdW = 1/n*g@X.T+2*Lambda*W
	dJdb = g@np.ones((n,1))/n
	return [dJdW, dJdb]

def computeGradient(X,Y,P,W,b,Lambda):
	n = np.shape(X)[1]
	g = -(Y-P)
	dJdW = g@X.T/n+2*Lambda*W
	dJdb = g@np.ones((n,1))/n
	return [dJdW, dJdb]

def ComputeGradsNumSlow(X, Y, W, b, Lambda, h):
    K = W.shape[0]
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((K, 1))

    for i in range(b.shape[0]):
        b_try = b.copy()
        b_try[i] -= h
        c1 = computeCost(X, Y, W, b_try, Lambda)
        b_try = b.copy()
        b_try[i] += h
        c2 = computeCost(X, Y, W, b_try, Lambda)
        grad_b[i] = (c2 - c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W.copy()
            W_try[i, j] -= h
            c1 = computeCost(X, Y, W_try, b, Lambda)
            W_try = W.copy()
            W_try[i, j] += h
            c2 = computeCost(X, Y, W_try, b, Lambda)
            grad_W[i, j] = (c2 - c1) / (2*h)

    return grad_W, grad_b
def compute_relative_error(grad_num, grad_anal, eps):
    error = np.abs(grad_num - grad_anal)/np.maximum(eps, np.abs(grad_num) + np.abs(grad_anal))
    return error

def acceptable_ratio(error,tolerance):
	return 100*np.sum(np.array(error)<tolerance)/np.size(error)
def compare_gradients(X_train,Y_train,Lambda,num_dim,num_samples):
	tolerances = [1e-1,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
	W,b = initializeWeights()
	P = evaluateClassifier(X_train[0:num_dim,0:num_samples],W[:,0:num_dim],b)
	grad_num = computeGradient(X_train[0:num_dim,0:num_samples],Y_train[:,0:num_samples],P,W[:,0:num_dim],Lambda)
	grad_anal = ComputeGradsNumSlow(X_train[0:num_dim,0:num_samples], Y_train[:,0:num_samples], W[:,0:num_dim], b, Lambda, 1e-6)
	error_W = compute_relative_error(grad_num[0], grad_anal[0], 1e-8)
	error_b = compute_relative_error(grad_num[1], grad_anal[1], 1e-8)
	ratios_W = []
	ratios_b = []
	for tol in tolerances:
		ratios_W.append(acceptable_ratio(error_W,tol))
		ratios_b.append(acceptable_ratio(error_b,tol))
	return ratios_W,ratios_b

def miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda):
	dim = np.shape(X_train)[0]
	n = np.shape(X_train)[1]
	labels = np.shape(Y_train)[0]
	cost_train = []
	cost_val = []
	tolerances = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
	W,b = initializeWeights()
	for epoch in range(epochs):
		randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
		currentX = X_train[:,randInd].reshape(dim,batch_size)
		currentY = Y_train[:,randInd].reshape(labels,batch_size)
		P = evaluateClassifier(currentX,W,b)
		[dJdW, dJdb] = computeGradient(currentX,currentY,P,W,Lambda)
		W -= eta*dJdW
		b -= eta*dJdb
		if epoch%(n//batch_size)==0:
			cost_train.append(computeCost(X_train,Y_train,W,b,Lambda))
			cost_val.append(computeCost(X_val,one_hot_encoding(y_val),W,b,Lambda))
			if len(cost_val)>=2:
				if (cost_val[-1]-cost_val[-2])>0:
					print("Early stopping at epoch: ", str(epoch))
					break
	test_acc = computeAccuracy(X_test,y_test,W,b)
	error = []
	error.append(compute_relative_error(computeGradient(X_train,Y_train,evaluateClassifier(X_train,W,b))[0], ComputeGradsNumSlow(X_train, Y_train, W, b, Lambda, 1e-6)[0], 1e-8)) 
	acceptable_ratio(error,tolerance)
	return cost_train,cost_val,test_acc,W

""" GRADIENT CHECK 
X_train, y_train, Y_train = load_batch(datapath+'data_batch_1')
X_val, y_val,Y_val = load_batch(datapath+'data_batch_2')
X_test, y_test, Y_test= load_batch(datapath+'data_batch_3')

X_train,X_val,X_test=preprocess(X_train,X_val,X_test)



Lambda = 0.1

ratios_w,ratios_b=compare_gradients(X_train,Y_train,Lambda,20,10)
tolerances = [1e-1,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
plt.semilogx(tolerances,ratios_w,'-o',tolerances,ratios_b,'-o')
plt.legend(["Ratio in W","Ratio in b"])
plt.ylabel('Ratio of correct analytical gradients')
plt.xlabel('Tolerance')
ax = plt.gca()
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
"""

""" PARAMETER STUDY 
# Lambda, epochs, batch_size, eta = 0, 40, 100, 0.1
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 0, 40, 100, 0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 0.1,40,100,0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 1,40,100,0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)


PLOTS OF TRAINING AND VALIDATION LOSS
# plt.plot(np.arange(len(cost_train)),cost_train,np.arange(len(cost_train)),cost_val)
# plt.legend(["Training loss","Validation loss"])
# #plt.axis([0,40,1.6,2.5])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# print(test_acc)
# plt.show()

VISUALIZATION OF WEIGHT MATRIX
# fig, axes = plt.subplots(1, 10, figsize=(25,5))
# for i in range(10):
# 	im_red = W[i,0:1024].reshape(32,32)
# 	im_green = W[i,1024:2048].reshape(32,32)
# 	im_blue = W[i,2048:3072].reshape(32,32)
# 	im = np.dstack((im_red, im_green, im_blue))
# 	im = (im-np.min(im))/(np.max(im)-np.min(im))	
# 	axes[i].imshow(im)
# 	axes[i].set_title(label_names[i])
# plt.show()

""" 




""" USING ALL AVAILABLE DATA
X_train1, y_train1, Y_train1 = load_batch(datapath+'data_batch_1')
X_train2, y_train2, Y_train2 = load_batch(datapath+'data_batch_2')
X_train3, y_train3, Y_train3 = load_batch(datapath+'data_batch_3')
X_train4, y_train4, Y_train4 = load_batch(datapath+'data_batch_4')
X_train,y_train,Y_train = np.concatenate((X_train1,X_train2,X_train3,X_train4)), np.concatenate((y_train1,y_train2,y_train3,y_train4)),np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4),axis=1)
X_testval, y_testval, Y_testval= load_batch(datapath+'data_batch_5')
X_val,y_val,X_test,y_test = X_testval[0:X_testval.shape[0]//2], y_testval[0:len(y_testval)//2], X_testval[X_testval.shape[0]//2:], y_testval[len(y_testval)//2:]
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train,X_val,X_test=preprocess(X_train,X_val,X_test)

"""

""" GRID SEARCH OF OPTIMAL HYPERPARAMETERS
batch_sizes = np.linspace(5,100,20)
etas = np.linspace(1e-5,1e-1,10)
Lambdas = np.linspace(0,10,100)
# for batch_size in batch_sizes:
# 	for eta in etas:
# 		for Lambda in Lambdas:
# 			counter += 1
# 			epochs = 40000//batch_size*20
# 			cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# 			test_accs[str(batch_size),str(eta),str(Lambda)] = test_acc
# 			print('Progress: %.1f' %counter/75)
# np.save('test_accs.npy',test_accs)
"""

"""SVM-loss implementation"""

# def miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda):
# 	dim = np.shape(X_train)[0]
# 	n = np.shape(X_train)[1]
# 	labels = np.shape(Y_train)[0]
# 	cost_train = []
# 	cost_val = []
# 	W,b = initializeWeights()
# 	epochs = epochs*n//batch_size
# 	for epoch in range(epochs):
# 		randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
# 		currentX = X_train[:,randInd].reshape(dim,batch_size)
# 		currentY = Y_train[:,randInd].reshape(labels,batch_size)
# 		P = evaluateClassifier(currentX,W,b)
# 		[dJdW, dJdb] = computeGradient_SVM(currentX,currentY,W,b,Lambda)
# 		W -= eta*dJdW
# 		b -= eta*dJdb
# 		if epoch%(n//batch_size)==0:
# 			cost_train.append(computeCost(X_train,Y_train,W,b,Lambda))
# 			cost_val.append(computeCost(X_val,one_hot_encoding(y_val),W,b,Lambda))
# 			if len(cost_val)>=2:
# 				if (cost_val[-1]-cost_val[-2])>0:
# 					print("Early stopping at epoch: ", str(epoch))
# 					break
# 	test_acc = computeAccuracy(X_test,y_test,W,b)
# 	return cost_train,cost_val,test_acc,W

# X_train, y_train, Y_train = load_batch(datapath+'data_batch_1')
# X_val, y_val,Y_val = load_batch(datapath+'data_batch_2')
# X_test, y_test, Y_test= load_batch(datapath+'data_batch_3')

# label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# X_train,X_val,X_test=preprocess(X_train,X_val,X_test)


# Lambda, epochs, batch_size, eta = 0, 40, 100, 0.1
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 0, 40, 100, 0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 0.1,40,100,0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
# Lambda, epochs, batch_size, eta = 1,40,100,0.001
# cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# print(test_acc)
"""