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


def computeGradient(X,Y,P,W,Lambda):
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
