import numpy as np
import matplotlib.pyplot as plt
import copy

datapath = 'C:\\Users\\arvid\\Documents\\KTH\\Masterkurser\\Deep Learning\\Assignments\\Assignment 3\\'


class k_layer_network():
	def __init__(self,input_dim,output_dim,nodes_list,init_method,Lambda):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.layers = self.set_layers(nodes_list,init_method)
		self.Lambda = Lambda
	class layer:
		def __init__(self,input_dim,output_dim,init_method):
			super().__init__()

			if init_method == 'he':
				self.W = np.random.normal(0,np.sqrt(2/(input_dim+output_dim)), size = (output_dim,input_dim))
				self.b = np.zeros((output_dim,1))

	def set_layers(self,nodes_list,init_method):
		dimensions = np.r_[self.input_dim,nodes_list,self.output_dim]
		nhidden = len(nodes_list)+1
		layers = []
		for i in range(nhidden):
			layers.append(self.layer(dimensions[i],dimensions[i+1],init_method))
		return layers
	def unpickle(self,file):
	    import pickle
	    with open(file, 'rb') as fo:
	        dict = pickle.load(fo, encoding='bytes')
	    return dict

	def one_hot_encoding(self,labels):
		t = np.concatenate((np.arange(len(labels)).reshape(len(labels),1),np.array(labels).reshape(len(labels),1)),axis=1)
		one_hot = np.zeros((len(labels),10))
		one_hot[t[:,0],t[:,1]] = 1
		return one_hot.T

	def load_batch(self,file):
		data_as_dict = self.unpickle(file)
		img_data = data_as_dict[b'data']
		labels = data_as_dict[b'labels']
		labels_encoded = self.one_hot_encoding(labels)
		return img_data, labels, labels_encoded

	def preprocess(self,tr_data,val_data,test_data):
		tr_mean = np.mean(tr_data,axis=0)
		tr_std = np.std(tr_data,axis=0)

		tr_data = (tr_data - tr_mean)/tr_std
		val_data = (val_data - tr_mean)/tr_std
		test_data = (test_data - tr_mean)/tr_std

		return tr_data.T, val_data.T, test_data.T
	def load_data(self,num_dims,num_samples):
		X_train1, y_train1, Y_train1 = self.load_batch(datapath+'data_batch_1')
		X_train2, y_train2, Y_train2 = self.load_batch(datapath+'data_batch_2')
		X_train3, y_train3, Y_train3 = self.load_batch(datapath+'data_batch_3')
		X_train4, y_train4, Y_train4 = self.load_batch(datapath+'data_batch_4')
		X_trainval, y_trainval, Y_trainval= self.load_batch(datapath+'data_batch_5')
		X_train,y_train,Y_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_trainval[0:len(X_trainval)//2])), np.concatenate((y_train1,y_train2,y_train3,y_train4,y_trainval[0:len(y_trainval)//2])),np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_trainval[:,0:Y_trainval.shape[1]//2]),axis=1)
		X_val,y_val,Y_val = X_trainval[X_trainval.shape[0]//2:], y_trainval[len(y_trainval)//2:],  Y_trainval[:,Y_trainval.shape[1]//2:]
		label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		X_test,y_test,Y_test = self.load_batch(datapath+'test_batch')
		self.X_train,self.X_val,self.X_test=self.preprocess(X_train,X_val,X_test)
		self.X_train = self.X_train[0:num_dims,0:num_samples]
		self.Y_train,self.y_train,self.Y_val,self.y_val,self.Y_test,self.y_test = \
		Y_train[:,0:num_samples],y_train[0:num_samples],Y_val,y_val,Y_test,y_test

	def forward_pass(self,X,Y):
		activations = [X]
		cost = 0
		for layer in self.layers:
			score = layer.W@X+layer.b
			X = score*(score>0)
			activations.append(X)
			cost += self.Lambda*np.sum(layer.W**2)
		P=np.exp(score)/np.sum(np.exp(score),axis=0)
		loss = np.sum(-np.log(np.multiply(Y, P).sum(axis=0))) / X.shape[1]
		cost += loss
		return P,activations,loss,cost
	def backward_pass(self,Y,P,X):
		G_batch = P-Y
		n_b = X[0].shape[1]
		gradW = []
		gradb = []
		k = len(self.layers)-1
		for i in range(k,-1,-1):
			dJdW = 1/n_b*G_batch@X[i].T+2*self.Lambda*self.layers[i].W
			dJdb = 1/n_b*G_batch@np.ones((n_b,1))
			gradW.append(dJdW)
			gradb.append(dJdb)
			G_batch = (self.layers[i].W).T@G_batch
			
			G_batch = G_batch*(X[i]>0)
		return gradW[::-1], gradb[::-1]
	def cyclic_rate(self,eta_min,eta_max,time,step):
		l=time//(2*step)
		if (time//step)%2==0:
			return eta_min+(time-2*l*step)*(eta_max-eta_min)/step
		else:
			return eta_max-(time-(2*l+1)*step)*(eta_max-eta_min)/step 
	def computeAccuracy(self,p,y):
		kstar = np.argmax(p,axis=0)
		acc = np.sum(kstar==y)/len(y)
		return acc
	def train(self,num_dim,num_samples,batch_size,epochs,eta_min,eta_max,step):
		self.load_data(num_dim,num_samples)
		n = np.shape(self.X_train)[1]
		labels = np.shape(self.Y_train)[0]
		for epoch in range(epochs):
			eta = self.cyclic_rate(eta_min,eta_max,epoch,step)
			randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
			currentX = self.X_train[:,randInd].reshape(num_dim,batch_size)
			currentY = self.Y_train[:,randInd].reshape(labels,batch_size)
			P,activations,loss,cost=self.forward_pass(currentX,currentY)
			gradW,gradb = self.backward_pass(currentY,P,activations)
			for i in range(len(self.layers)):
				self.layers[i].W -= eta*gradW[i]
				self.layers[i].b -= eta*gradb[i]
		P,_,_,_=self.forward_pass(self.X_test,self.Y_test)
		print(self.computeAccuracy(P,self.y_test))
		#self.compute_grads_num(self.X_train,self.Y_train,1e-6)
	def compute_grads_num(self, X, Y, h):
	    grad_W = []
	    grad_b = []
	    for idx, layer in enumerate(self.layers):
	        W = copy.deepcopy(layer.W)
	        b = copy.deepcopy(layer.b)
	        gradW = np.zeros(W.shape)
	        gradb = np.zeros(b.shape)
	        _, _, _,cost = self.forward_pass(X, Y)

	        for i in range(b.shape[0]):
	            self.layers[idx].b[i] += h
	            _, _, _, c2 = self.forward_pass(X, Y)
	            gradb[i] = (c2 - cost)/h
	            self.layers[idx].b[i] -= h

	        for i in range(W.shape[0]):
	            for j in range(W.shape[1]):
	                self.layers[idx].W[i, j] += h
	                _, _, _, c2 = self.forward_pass(X, Y)
	                gradW[i, j] = (c2 - cost)/h
	                self.layers[idx].W[i, j] -= h

	        grad_W.append(gradW)
	        grad_b.append(gradb)
	    return grad_W, grad_b

a=k_layer_network(3072,10,[50,50],'he',0.005)

a.train(3072,45000,100,4*5*450,1e-5,1e-1,5*450)
