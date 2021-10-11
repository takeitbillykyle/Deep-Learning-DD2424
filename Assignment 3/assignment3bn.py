import numpy as np
import matplotlib.pyplot as plt
import copy
import sys

epsilon = sys.float_info.epsilon

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
				self.gamma = np.ones((output_dim,1))
				self.beta = np.zeros((output_dim,1))

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

	def batch_norm_backward(self,G_batch,S_batch,mu,var_sqrt):
		n = G_batch.shape[1]
		mu = mu.reshape(mu.shape[0],1)
		sigma1 = 1/(var_sqrt+epsilon)
		sigma2 = 1/(var_sqrt+epsilon)**3
		sigma1=sigma1.reshape(sigma1.shape[0],1)
		sigma2=sigma2.reshape(sigma2.shape[0],1)
		
		G1 = G_batch*(sigma1@np.ones((1,n)))
		G2 = G_batch*(sigma2@np.ones((1,n)))

		D = S_batch-mu@np.ones((1,n))
		c = (G2*D)@np.ones((n,1))
		G_batch = G1-1/n*(G1@np.ones((n,1)))@np.ones((1,n))-1/n*D*(c@np.ones((1,n)))
		return G_batch
	def forward_pass(self,X,Y,batch_norm,test_time,mu_av=None,var_av=None,first_time=True):
		activations = [X]
		S_hat = []
		scores = []
		mus = []
		var_sqrts = []
		mu_avs = []
		var_avs = []
		cost = 0
		alpha=0.9
		if not batch_norm:
			for layer in self.layers:
				score = layer.W@X+layer.b
				X = score*(score>0)
				activations.append(X)
				cost += self.Lambda*np.sum(layer.W**2)
		elif batch_norm and not test_time:
			for i,layer in enumerate(self.layers):
				score = layer.W@X+layer.b
				scores.append(score)
				mu,var = np.mean(score,axis=1),np.var(score,axis=1)
				var_sqrt = np.sqrt(var)
				mus.append(mu)
				var_sqrts.append(var_sqrt)
				if first_time:
					mu_avs.append(mu)
					var_avs.append(var_sqrt)
				else:
					mu_av[i] = alpha*mu_av[i] + (1-alpha)*mu
					var_av[i] = alpha*var_av[i]+(1-alpha)*var_sqrt
				s_hat = np.diag(1/var_sqrt+epsilon)@(score-mu.reshape(mu.shape[0],1))
				S_hat.append(s_hat)
				
				s_tilde = layer.gamma*s_hat+layer.beta
				X = s_tilde*(s_tilde>0)
				activations.append(X)
				cost += self.Lambda*np.sum(layer.W**2)
			if first_time:
				mu_av = mu_avs
				var_av = var_avs
		elif batch_norm and test_time:
			for i,layer in enumerate(self.layers):
				score = layer.W@X+layer.b
				s_hat = np.linalg.inv(np.diag(var_av[i]+epsilon))@(score-mu_av[i].reshape(mu_av[i].shape[0],1))
				s_tilde = np.repeat(layer.gamma,s_hat.shape[1],axis=1)*s_hat+layer.beta
				X = s_tilde*(s_tilde>0)
			return np.exp(score)/np.sum(np.exp(score),axis=0),None,None,None,None,None,None,None,None,None
		P=np.exp(score)/np.sum(np.exp(score),axis=0)
		loss =-1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@P+epsilon)))
		cost += loss
		return P,activations,loss,cost,mus,var_sqrts,mu_av,var_av,S_hat,scores
	def backward_pass(self,Y,P,X,S_hat=None,S_batch=None,mu=None,var_sqrt=None):
		G_batch = P-Y
		n_b = X[0].shape[1]
		gradW = []
		gradb = []
		gradgamma = []
		gradbeta = []
		k = len(self.layers)-1
		print(k)
		for i in range(k,-1,-1):
			dJdgamma = 1/n_b*(G_batch*S_hat[i])@np.ones((n_b,1))
			dJdbeta = 1/n_b*G_batch@np.ones((n_b,1))

			G_batch = G_batch*(self.layers[i].gamma@np.ones((1,n_b)))
			G_batch = self.batch_norm_backward(G_batch,S_batch[i],mu[i],var_sqrt[i])
			dJdW = 1/n_b*G_batch@X[i].T+2*self.Lambda*self.layers[i].W
			dJdb = 1/n_b*G_batch@np.ones((n_b,1))
			gradW.append(dJdW)
			gradb.append(dJdb)
			gradgamma.append(dJdgamma)
			gradbeta.append(dJdbeta)
			G_batch = (self.layers[i].W).T@G_batch
			G_batch = G_batch*(X[i]>0)
		return gradW[::-1], gradb[::-1], gradgamma[::-1], gradbeta[::-1]
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
		test_acc = []
		for epoch in range(epochs):
			eta = self.cyclic_rate(eta_min,eta_max,epoch,step)
			randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
			currentX = self.X_train[:,randInd].reshape(num_dim,batch_size)
			currentY = self.Y_train[:,randInd].reshape(labels,batch_size)
			if epoch==0:	
				P,activations,loss,cost,mus,var_sqrts,mu_av,var_av,S_hat,S_batch=self.forward_pass(currentX,currentY,True,False)
			else:
				P,activations,loss,cost,mus,var_sqrts,mu_av,var_av,S_hat,S_batch=self.forward_pass(currentX,currentY,True,False,mu_av,var_av,False)
			gradW,gradb,gradgamma,gradbeta = self.backward_pass(currentY,P,activations,S_hat,S_batch,mus,var_sqrts)
			for i in range(len(self.layers)):
				self.layers[i].W -= eta*gradW[i]
				self.layers[i].b -= eta*gradb[i]
				self.layers[i].gamma -= eta*gradgamma[i]
				self.layers[i].beta -= eta*gradbeta[i]

		#P,_,_,_,_,_,_,_=self.forward_pass(self.X_train[:,0:100],self.Y_test[:,0:100],True,False)
			if epoch%500==0:
				P,_,_,_,_,_,_,_,_,_=self.forward_pass(self.X_test,self.Y_test,True,True,mu_av,var_av,False)
				test_acc.append(self.computeAccuracy(P,self.y_test))
		#print(self.computeAccuracy(P,self.y_train[0:100]))
		#self.compute_grads_num(self.X_train,self.Y_train,1e-6)
		return test_acc
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

a=k_layer_network(3072,10,[50,30,20,20,10,10,10,10],'he',0.005)
#4*5*450
test_acc = a.train(3072,45000,100,10,1e-5,1e-1,5*450)
print(test_acc)
plt.plot(np.arange(0,len(test_acc),1),test_acc)
plt.show()