import numpy as np
import matplotlib.pyplot as plt
import copy
import imgaug.augmenters as iaa
import imgaug as ia

datapath = 'C:\\Users\\arvid\\Documents\\KTH\\Masterkurser\\Deep Learning\\Assignments\\Assignment 2\\'


np.random.seed(100)
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

def initializeWeights(m,d,K):
	W1 = np.random.normal(0,1/np.sqrt(d),size=m*d).reshape(m,d)
	W2 = np.random.normal(0,1/np.sqrt(m),size=K*m).reshape(K,m)
	b1 = np.zeros((m,1))
	b2 = np.zeros((K,1))

	return W1,W2,b1,b2

def softMax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)

def evaluateClassifier(X,W1,W2,b1,b2):
	"""
	 W1 = m x d
	 w2 = K x d
	 X = d x N 
	 b1 = m x 1
	 b2 = K x1
	"""
	s1 = W1@X+b1
	h = s1*(s1>0)
	s = W2@h + b2
	return softMax(s),h

def computeCost(X,Y,W1,W2,b1,b2,Lambda,return_loss = False):
	p = evaluateClassifier(X,W1,W2,b1,b2)[0]
	loss = --1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@p)))
	J = Lambda*(np.sum(W1**2)+np.sum(W2**2))+loss
	if return_loss:
		return J,loss
	else:
		return J

def computeAccuracy(X,y,W1,W2,b1,b2):
	p = evaluateClassifier(X,W1,W2,b1,b2)[0]
	kstar = np.argmax(p,axis=0)
	acc = np.sum(kstar==y)/len(y)
	return acc

def computeGradient(X,Y,P,h,W1,W2,b1,b2,Lambda):
	n = np.shape(X)[1]
	G = -(Y-P)
	dJdW2 = G@h.T/n+2*Lambda*W2
	dJdb2 = G@np.ones((n,1))/n
	G = W2.T@G
	G = G*(h>0)
	dJdW1 = G@X.T/n+2*Lambda*W1
	dJdb1 = G@np.ones((n,1))/n
	return [dJdW1,dJdW2,dJdb1,dJdb2]

def ComputeGradsNumSlow(X, Y, W1,W2, b1,b2, Lambda, h):
    W = [W1,W2]
    b = [b1,b2]
    grad_W = [np.zeros(W1.shape),np.zeros(W2.shape)]
    grad_b = [np.zeros(b1.shape),np.zeros(b2.shape)]
    for i in range(len(b)):
        grad_b[i] = np.zeros(b[i].shape)
        for j in range(len(b[i])):
            b_try = copy.deepcopy(b)
            b_try[i][j] -= h
            c1 = computeCost(X, Y, W1,W2, b_try[0],b_try[1], Lambda)
            b_try = copy.deepcopy(b)
            b_try[i][j] += h
            c2 = computeCost(X, Y, W1,W2, b_try[0],b_try[1], Lambda)
            grad_b[i][j] = (c2 - c1)/(2*h)
    for i in range(len(W)):
        grad_W[i] = np.zeros(W[i].shape)
        for j in range(W[i].shape[0]):
            for k in range(W[i].shape[1]):
                W_try = copy.deepcopy(W)
                W_try[i][j,k] -= h
                c1 = computeCost(X, Y, W_try[0],W_try[1], b1,b2, Lambda)
                W_try = copy.deepcopy(W)
                W_try[i][j,k] += h
                c2 = computeCost(X, Y, W_try[0],W_try[1], b1,b2, Lambda)
                grad_W[i][j,k] = (c2 - c1)/(2*h)  
    return grad_W, grad_b
def compute_relative_error(grad_num, grad_anal, eps):
    error = np.abs(grad_num - grad_anal)/np.maximum(eps, np.abs(grad_num) + np.abs(grad_anal))
    return error

def acceptable_ratio(error,tolerance):
	return 100*np.sum(np.array(error)<tolerance)/np.size(error)
def compare_gradients(X_train,Y_train,Lambda,num_dim,num_samples):
	tolerances = [1e-1,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
	W1,W2,b1,b2 = initializeWeights(50,num_dim,10)
	P,h = evaluateClassifier(X_train[0:num_dim,0:num_samples],W1[:,0:num_dim],W2,b1,b2)
	[dJdW1,dJdW2,dJdb1,dJdb2] = computeGradient(X_train[0:num_dim,0:num_samples],Y_train[:,0:num_samples],P,h,W1[:,0:num_dim],W2,b1,b2,Lambda)
	grad_W, grad_b = ComputeGradsNumSlow(X_train[0:num_dim,0:num_samples], Y_train[:,0:num_samples],W1[:,0:num_dim],W2,b1,b2, Lambda, 1e-6)
	error_W1 = compute_relative_error(dJdW1, grad_W[0], 1e-8)
	error_W2 = compute_relative_error(dJdW2, grad_W[1], 1e-8)
	error_b1 = compute_relative_error(dJdb1, grad_b[0], 1e-8)
	error_b2 = compute_relative_error(dJdb2, grad_b[1], 1e-8)
	ratios_W1 = []
	ratios_W2 = []
	ratios_b1 = []
	ratios_b2 = []
	for tol in tolerances:
		ratios_W1.append(acceptable_ratio(error_W1,tol))
		ratios_W2.append(acceptable_ratio(error_W2,tol))
		ratios_b1.append(acceptable_ratio(error_b1,tol))
		ratios_b2.append(acceptable_ratio(error_b2,tol))
	return ratios_W1,ratios_W2,ratios_b1,ratios_b2
def cyclic_rate(eta_min,eta_max,time,step):
	l=time//(2*step)
	if (time//step)%2==0:
		return eta_min+(time-2*l*step)*(eta_max-eta_min)/step
	else:
		return eta_max-(time-(2*l+1)*step)*(eta_max-eta_min)/step 
def cyclic_rate_sin(eta_min,eta_max,time,step):
	return eta_min+(eta_max-eta_min)*np.abs(np.sin(0.5*np.pi*time/step))

def miniBatchGD(X_train,Y_train,X_val,Y_val,y_val,X_test,Y_test,y_test,batch_size,epochs,Lambda,eta_min,eta_max,step,nodes,dropout_percent):
	dim = np.shape(X_train)[0]
	n = np.shape(X_train)[1]
	labels = np.shape(Y_train)[0]
	cost_train = []
	cost_val = []
	acc_train = []
	acc_val = []
	loss_train = []
	loss_val = []
	W1,W2,b1,b2 = initializeWeights(nodes,3072,10)
	dropout_percent1 = dropout_percent
	dropout_percent2 = dropout_percent
	test_acc = []
	for epoch in range(epochs):
		eta = cyclic_rate(eta_min,eta_max,epoch,step)
		randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
		currentX = X_train[:,randInd].reshape(dim,batch_size)
		currentY = Y_train[:,randInd].reshape(labels,batch_size)
		W1temp = W1*(np.ones(W1.shape)*np.random.binomial([np.ones((300,1))],1-dropout_percent1)[0]/(1-dropout_percent1))
		W2temp =  W2.T*(np.ones(W2.T.shape)*np.random.binomial([np.ones((300,1))],1-dropout_percent2)[0]/(1-dropout_percent2))
		W2temp=W2temp.T
		P,h = evaluateClassifier(currentX,W1temp,W2temp,b1,b2)
		[dJdW1,dJdW2,dJdb1,dJdb2] = computeGradient(currentX,currentY,P,h,W1temp,W2temp,b1,b2,Lambda)
		W1 -= eta*dJdW1
		W2 -= eta*dJdW2
		b1 -= eta*dJdb1
		b2 -= eta*dJdb2
		#if epoch%batch_size==0:
		#	acc_val.append(computeAccuracy(X_val,y_val,W1,W2,b1,b2))
		# 	train_cost,train_loss = computeCost(X_train,Y_train,W1,W2,b1,b2,Lambda,True)
		# 	val_cost,val_loss = computeCost(X_val,Y_val,W1,W2,b1,b2,Lambda,True)
		# 	cost_train.append(train_cost)
		# 	cost_val.append(val_cost)
		# 	loss_train.append(train_loss)
		# 	loss_val.append(val_loss)

		
	#,cost_train,cost_val,loss_train,loss_val
	return computeAccuracy(X_test,y_test,W1,W2,b1,b2)
	#return cost_train,cost_val,acc_train,acc_val


X_train1, y_train1, Y_train1 = load_batch(datapath+'data_batch_1')
X_train2, y_train2, Y_train2 = load_batch(datapath+'data_batch_2')
X_train3, y_train3, Y_train3 = load_batch(datapath+'data_batch_3')
X_train4, y_train4, Y_train4 = load_batch(datapath+'data_batch_4')
X_trainval, y_trainval, Y_trainval= load_batch(datapath+'data_batch_5')
X_train,y_train,Y_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_trainval[0:len(X_trainval)//2])), np.concatenate((y_train1,y_train2,y_train3,y_train4,y_trainval[0:len(y_trainval)//2])),np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_trainval[:,0:Y_trainval.shape[1]//2]),axis=1)
X_val,y_val,Y_val = X_trainval[X_trainval.shape[0]//2:], y_trainval[len(y_trainval)//2:],  Y_trainval[:,Y_trainval.shape[1]//2:]
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_test,y_test,Y_test = load_batch(datapath+'test_batch')
X_train,X_val,X_test=preprocess(X_train,X_val,X_test)

noSamples = 45000
X_train = X_train[:,0:noSamples]
y_train = y_train[0:noSamples]
Y_train = Y_train[:,0:noSamples]


batch_size = 160
Lambda = 1e-4
eta_min = 1e-5
eta_max = 1e-1
step= 1100
nodes = 300
nocycles = 8
epochs = step*2*nocycles
dps = [0,0.1,0.2,0.3,0.4,0.5,0.6]
for dp in dps:
	test=miniBatchGD(X_train,Y_train,X_val,Y_val,y_val,X_test,Y_test,y_test,batch_size,epochs,Lambda,eta_min,eta_max,step,nodes,dp)
	print("DROPOUT PERCENTAGE: ", str(100*dp), ' ACCURACY: ', str(test))