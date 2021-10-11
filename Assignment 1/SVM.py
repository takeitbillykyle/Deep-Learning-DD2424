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

def computeCost(X,Y,W,b,Lambda):
	p = evaluateClassifier(X,W,b)
	J = Lambda*np.sum(W**2)-1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@p)))
	return J

def computeAccuracy(X,y,W,b):
	p = evaluateClassifier(X,W,b)
	kstar = np.argmax(p,axis=0)
	acc = np.sum(kstar==y)/len(y)
	return acc

def computeGradient_SVM(X,Y,W,b,Lambda):
	n = np.shape(X)[1]
	s = evaluateClassifier(X,W,b)
	scores=np.sum(s*Y,axis=0)
	margins = s-scores+1
	ind = margins>0
	G = ind-Y
	sumG=np.sum(G,axis=0)
	g=np.where(Y!=1,G,-sumG)
	dJdW = 1/n*g@X.T+2*Lambda*W
	dJdb = g@np.ones((n,1))/n
	return [dJdW, dJdb]

def computeGradient(X,Y,W,b,Lambda):
	n = np.shape(X)[1]
	g = -(Y-P)
	dJdW = g@X.T/n+2*Lambda*W
	dJdb = g@np.ones((n,1))/n
	return [dJdW, dJdb]

def ComputeGradsNumSlow(X, Y, W, b, penalty, h):
    K = W.shape[0]
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((K, 1))

    for i in range(b.shape[0]):
        b_try = b.copy()
        b_try[i] -= h
        c1 = computeCost(X, Y, W, b_try, penalty)
        b_try = b.copy()
        b_try[i] += h
        c2 = computeCost(X, Y, W, b_try, penalty)
        grad_b[i] = (c2 - c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W.copy()
            W_try[i, j] -= h
            c1 = computeCost(X, Y, W_try, b, penalty)
            W_try = W.copy()
            W_try[i, j] += h
            c2 = computeCost(X, Y, W_try, b, penalty)
            grad_W[i, j] = (c2 - c1) / (2*h)

    return grad_W, grad_b
def compute_relative_error_gradients(grad_num, grad_anal, eps):
    err = np.abs(grad_num - grad_anal) / np.maximum(eps*np.ones(grad_num.shape), np.abs(grad_num) + np.abs(grad_anal))
    return err

def miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda):
	dim = np.shape(X_train)[0]
	n = np.shape(X_train)[1]
	labels = np.shape(Y_train)[0]
	cost_train = []
	cost_val = []
	W,b = initializeWeights()
	epochs = epochs*n//batch_size
	for epoch in range(epochs):
		randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
		currentX = X_train[:,randInd].reshape(dim,batch_size)
		currentY = Y_train[:,randInd].reshape(labels,batch_size)
		P = evaluateClassifier(currentX,W,b)
		[dJdW, dJdb] = computeGradient_SVM(currentX,currentY,W,b,Lambda)
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
	return cost_train,cost_val,test_acc,W

X_train, y_train, Y_train = load_batch(datapath+'data_batch_1')
X_val, y_val,Y_val = load_batch(datapath+'data_batch_2')
X_test, y_test, Y_test= load_batch(datapath+'data_batch_3')

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train,X_val,X_test=preprocess(X_train,X_val,X_test)


Lambda, epochs, batch_size, eta = 0, 40, 100, 0.1
cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
print(test_acc)
Lambda, epochs, batch_size, eta = 0, 40, 100, 0.001
cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
print(test_acc)
Lambda, epochs, batch_size, eta = 0.1,40,100,0.001
cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
print(test_acc)
Lambda, epochs, batch_size, eta = 1,40,100,0.001
cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
print(test_acc)






# batch_sizes = [5,10,25,50]
# etas = np.linspace(1e-4,1e-2,5)
# Lambdas = np.linspace(0,2,5)


# best_test = 0
# for batch_size in batch_sizes:
# 	for eta in etas:
# 		for Lambda in Lambdas:
# 			epochs = 40000//batch_size*20
# 			cost_train,cost_val,test_acc,W=miniBatchGD(X_train,Y_train,X_val,y_val,X_test,y_test,batch_size,eta,epochs,Lambda)
# 			if test_acc>best_test:
# 				best_test = test_acc
# 			print("Current best: " + str(test_acc))
# 			print("Settings: ", "batch: " + str(batch_size) + "eta: " + str(eta)+"lambda: "+str(Lambda))
	
