import numpy as np
import matplotlib.pyplot as plt
import copy
import sys

epsilon = sys.float_info.epsilon

datapath = 'C:\\Users\\arvid\\Documents\\KTH\\Masterkurser\\Deep Learning\\Assignments\\Assignment 3\\'

class k_layer_network():
	def __init__(self,input_dim,output_dim,nodes_list,init_method,Lambda,Sigma = 0):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.layers = self.set_layers(nodes_list,init_method,Sigma)
		self.Lambda = Lambda
	class layer:
		def __init__(self,input_dim,output_dim,init_method,Sigma):
			super().__init__()
			if init_method == 'he':
				self.W = np.random.normal(0,np.sqrt(2/(input_dim+output_dim)), size = (output_dim,input_dim))
				self.b = np.zeros((output_dim,1))
				self.gamma = np.ones((output_dim,1))
				self.beta = np.zeros((output_dim,1))
				self.mu_av = np.zeros((output_dim,1))
				self.var_av = np.zeros((output_dim,1))
				self.mu = np.zeros((output_dim,1))
				self.var = np.zeros((output_dim,1))
			elif init_method == 'normal':
				self.W = np.random.normal(0,Sigma, size = (output_dim,input_dim))
				self.b = np.zeros((output_dim,1))
				self.gamma = np.ones((output_dim,1))
				self.beta = np.zeros((output_dim,1))
				self.mu_av = np.zeros((output_dim,1))
				self.var_av = np.zeros((output_dim,1))
				self.mu = np.zeros((output_dim,1))
				self.var = np.zeros((output_dim,1))
	def set_layers(self,nodes_list,init_method,Sigma):
		dimensions = np.r_[self.input_dim,nodes_list,self.output_dim]
		nhidden = len(nodes_list)+1
		layers = []
		for i in range(nhidden):
			layers.append(self.layer(dimensions[i],dimensions[i+1],init_method,Sigma))
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

	

	def forward_pass(self,X,Y,testing = False, first_time = False,batch_norm = True):

		alpha = 0.9
		cost = 0

		S_batch = []
		S_hat = []

		activations = [X]
		if not batch_norm:
			for layer in self.layers:
				score = layer.W@X+layer.b
				X = score*(score>0)
				activations.append(X)
				cost += self.Lambda*np.sum(layer.W**2)
			P = np.exp(score) / np.sum(np.exp(score), axis = 0)
			loss =-1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@P+epsilon)))
			cost += loss
			return P,activations,loss,cost
		else:
			if not testing:
				for i, layer in enumerate(self.layers):
					score = layer.W@X + layer.b
					S_batch.append(score)
					mu = np.mean(score, axis = 1)
					var = np.var(score, axis = 1)

					#BatchNormalize

					s_hat = np.diag(1/np.sqrt(var+epsilon))@(score - mu.reshape(len(mu),1))
					S_hat.append(s_hat)
					s_tilde = np.tile(layer.gamma,s_hat.shape[1])*s_hat + layer.beta

					X = s_tilde * (s_tilde > 0)
					activations.append(X)

					if first_time:

						layer.mu_av = mu
						layer.var_av = var

					else:

						layer.mu_av = alpha*layer.mu_av + (1-alpha)*mu
						layer.var_av = alpha*layer.var_av + (1-alpha)*var

					layer.mu = np.array(mu).reshape(len(mu),1)
					layer.var = np.array(var).reshape(len(var),1)

					cost += self.Lambda*np.sum(layer.W**2)

			if testing:
				for i, layer in enumerate(self.layers):
					score = layer.W@X + layer.b
					#BatchNormalize
					s_hat = np.diag(1/np.sqrt(layer.var_av+epsilon))@(score - layer.mu_av.reshape(len(layer.mu_av),1))
					s_tilde = np.tile(layer.gamma,s_hat.shape[1])*s_hat + layer.beta
					X = s_tilde * (s_tilde > 0)
		P = np.exp(score) / np.sum(np.exp(score), axis = 0)

		loss =-1/(np.shape(X)[1])*np.sum(np.log(np.diag(Y.T@P+epsilon)))
		cost += loss
		return P,activations,S_hat,S_batch, loss,cost



	def batch_norm_backward(self,G_batch,S_batch,mu,var):
		n = G_batch.shape[1]
		sigma1 = (1/(var + epsilon)**(0.5)).reshape(len(var),1)
		sigma2 = (1/(var + epsilon)**(1.5)).reshape(len(var),1)

		G1 = G_batch * (sigma1@np.ones((1,n)))
		G2 = G_batch * (sigma2@np.ones((1,n)))

		D = S_batch - mu@np.ones((1,n))
		c = (G2*D)@np.ones((n,1))
		G_batch = G1 - 1/n*(G1@np.ones((n,1)))@np.ones((1,n)) - 1/n*D*(c@np.ones((1,n)))

		return G_batch


	def backward_pass(self,Y,P,X,S_hat,S_batch,batch_norm=True):
		G_batch = P-Y
		n = X[0].shape[1]
		gradW = []
		gradb = []
		

		gradgamma = []
		gradbeta = []
		k = len(self.layers)-1
		if not batch_norm:
			for i in range(k,-1,-1):
				dJdW = 1/n*G_batch@X[i].T+2*self.Lambda*self.layers[i].W
				dJdb = 1/n*G_batch@np.ones((n,1))
				gradW.append(dJdW)
				gradb.append(dJdb)
				G_batch = (self.layers[i].W).T@G_batch
				G_batch = G_batch*(X[i]>0)
			return gradW[::-1], gradb[::-1]
		gradW.append(1/n*G_batch@X[k].T+2*self.Lambda*self.layers[k].W)
		gradb.append(1/n*G_batch@np.ones((n,1)))
		G_batch = self.layers[k].W.T@G_batch
		G_batch = G_batch*(X[k]>0)
		for i in range(k-1,-1,-1):
			dJdgamma = 1/n * (G_batch * S_hat[i])@np.ones((n,1))
			dJdbeta = 1/n * G_batch@np.ones((n,1))

			G_batch = G_batch*(self.layers[i].gamma@np.ones((1,n)))

			G_batch = self.batch_norm_backward(G_batch,S_batch[i],self.layers[i].mu,self.layers[i].var)

			dJdW = 1/n*G_batch@X[i].T + 2*self.Lambda*self.layers[i].W
			dJdb = 1/n*G_batch@np.ones((n,1))

			gradW.append(dJdW)
			gradb.append(dJdb)
			gradgamma.append(dJdgamma)
			gradbeta.append(dJdbeta)
			
			G_batch = self.layers[i].W.T@G_batch
			G_batch = G_batch * (X[i]>0)
		return gradW[::-1],gradb[::-1],gradgamma[::-1],gradbeta[::-1]
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
	def train(self,num_dim,num_samples,batch_size,epochs,eta_min,eta_max,step,batch_norm = True):
		self.load_data(num_dim,num_samples)
		n = np.shape(self.X_train)[1]
		labels = np.shape(self.Y_train)[0]
		train_loss = []
		val_loss = []
		train_acc = []
		val_acc = []
		final_test=0
		for epoch in range(epochs):
			eta = self.cyclic_rate(eta_min,eta_max,epoch,step)
			randInd = np.random.choice(range(n),size = (batch_size,1),replace=False)
			currentX = self.X_train[:,randInd].reshape(num_dim,batch_size)
			currentY = self.Y_train[:,randInd].reshape(labels,batch_size)
			if not batch_norm:
				P,activations,loss,cost=self.forward_pass(currentX,currentY,False,False,False)
				gradW,gradb = self.backward_pass(currentY,P,activations,None,None,False)
				for i in range(len(self.layers)):
					self.layers[i].W -= eta*gradW[i]
					self.layers[i].b -= eta*gradb[i]
				if epoch%500==0:
					P_val,activations_val,loss_val,cost_val=self.forward_pass(self.X_val,self.Y_val,False,False,False)
					P,activations,loss,cost=self.forward_pass(self.X_train[:,0:1000],self.Y_train[:,0:1000],False,False,False)
					val_loss.append(loss_val)
					train_loss.append(loss)
					val_acc.append(self.computeAccuracy(P_val,self.y_val))
					train_acc.append(self.computeAccuracy(P,self.y_train[0:1000]))
				if epoch==epochs-1:
					P_test,activations_test,loss_test,cost_test=self.forward_pass(self.X_test,self.Y_test,False,False,False)
					final_test = self.computeAccuracy(P_test,self.y_test)
			else:
				if epoch==0:	
					P,activations,S_hat,S_batch,loss,cost=self.forward_pass(currentX,currentY,False,True)
				else:
					P,activations,S_hat,S_batch,loss,cost=self.forward_pass(currentX,currentY,False,False)
		
				gradW,gradb,gradgamma,gradbeta = self.backward_pass(currentY,P,activations,S_hat,S_batch)
				for i in range(len(self.layers)):
					self.layers[i].W -= eta*gradW[i]
					self.layers[i].b -= eta*gradb[i]
					if i!=len(self.layers)-1:
						self.layers[i].gamma -= eta*gradgamma[i]
						self.layers[i].beta -= eta*gradbeta[i]

				if epoch%500==0:
					P_val,activations_val,S_hat_val,S_batch_val, loss_val,cost_val=self.forward_pass(self.X_val,self.Y_val,True,False)
					P,activations,S_hat,S_batch, loss,cost=self.forward_pass(self.X_train[:,0:1000],self.Y_train[:,0:1000],True,False)
					val_acc.append(self.computeAccuracy(P_val,self.y_val))
					val_loss.append(loss_val)
					train_loss.append(loss)
					train_acc.append(self.computeAccuracy(P,self.y_train[0:1000]))
				if epoch==epochs-1:
					P_test,activations_test,S_hat_test,S_batch_test, loss_test,cost_test=self.forward_pass(self.X_test,self.Y_test,True,False)
					final_test = self.computeAccuracy(P_test,self.y_test)
		return train_loss,train_acc,val_loss,val_acc,final_test
	def acceptable_ratio(self,error,tolerance):
		return 100*np.sum(np.array(error)<tolerance)/np.size(error)

	def compute_relative_error(self,grad_num, grad_anal, eps):
		error = np.abs(grad_num - grad_anal)/np.maximum(eps, np.abs(grad_num) + np.abs(grad_anal))
		return error
	def compare_grads(self,num_dim,num_samples):
		self.load_data(num_dim,num_samples)
		tolerances = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
		n = np.shape(self.X_train)[1]
		labels = np.shape(self.Y_train)[0]
		randInd = np.random.choice(range(n),size = (num_samples,1),replace=False)
		currentX = self.X_train[:,randInd].reshape(num_dim,num_samples)
		currentY = self.Y_train[:,randInd].reshape(labels,num_samples)
		#forward_pass(self,X,Y,testing = False, first_time = False
		testing = False
		first_time= True
		P,activations,S_hat,S_batch, loss,cost=self.forward_pass(currentX,currentY,testing,first_time)
		gradW,gradb,gradgamma,gradbeta = self.backward_pass(currentY,P,activations,S_hat,S_batch)
		gradW_num,gradb_num,gradgamma_num,gradbeta_num = self.compute_grads_num(currentX,currentY,1e-5)
		ratios_W = {}
		ratios_b = {}
		ratios_gamma = {}
		ratios_beta = {}
		error_W = []
		error_b = []
		error_gamma = []
		error_beta =[]
		for i in range(len(gradW)):
			error_W.append(self.compute_relative_error(gradW_num[i], gradW[i], 1e-8))
			error_b.append(self.compute_relative_error(gradb_num[i], gradb[i], 1e-8))
		for i in range(len(gradbeta)):
			error_gamma.append(self.compute_relative_error(gradgamma_num[i], gradgamma[i], 1e-8))
			error_beta.append(self.compute_relative_error(gradbeta_num[i], gradbeta[i], 1e-8))
		for tol in tolerances:
			for i in range(len(error_W)):
				ratios_W[str(tol),str(i+1)]=self.acceptable_ratio(error_W[i],tol)
				ratios_b[str(tol),str(i+1)]=self.acceptable_ratio(error_b[i],tol)
			for i in range(len(error_gamma)):
				ratios_gamma[str(tol),str(i+1)]=self.acceptable_ratio(error_gamma[i],tol)
				ratios_beta[str(tol),str(i+1)]=self.acceptable_ratio(error_beta[i],tol)
		

		return ratios_W,ratios_b,ratios_gamma,ratios_beta
	def compute_grads_num(self, X, Y, h):
	    grad_W = []
	    grad_b = []
	    grad_gamma = []
	    grad_beta = []
	    for idx, layer in enumerate(self.layers):
	        W = copy.deepcopy(layer.W)
	        b = copy.deepcopy(layer.b)
	        gradW = np.zeros(W.shape)
	        gradb = np.zeros(b.shape)
	        _,_,_, _, _,cost = self.forward_pass(X, Y)
	        
	        for i in range(b.shape[0]):
	            self.layers[idx].b[i] += h
	            _,_,_, _, _,c2 = self.forward_pass(X, Y)
	            gradb[i] = (c2 - cost)/h
	            self.layers[idx].b[i] -= h

	        for i in range(W.shape[0]):
	            for j in range(W.shape[1]):
	                self.layers[idx].W[i, j] += h
	                _,_,_, _, _,c2 = self.forward_pass(X, Y)
	                gradW[i, j] = (c2 - cost)/h
	                self.layers[idx].W[i, j] -= h

	        grad_W.append(gradW)
	        grad_b.append(gradb)
	        gamma = layer.gamma
	        beta = layer.beta
	        gradGamma = np.zeros(gamma.shape)
	        gradBeta = np.zeros(beta.shape)

	        for i in range(gamma.shape[0]):
	            gamma[i,:] -= h
	            _,_, _, _, _, c1 = self.forward_pass(X, Y)
	            gamma[i, :] += 2*h
	            _,_, _, _, _, c2 = self.forward_pass(X, Y)
	            gradGamma[i, :] = (c2-c1) / (2*h)
	            gamma[i, :] -= h
	        
	        for i in range(beta.shape[0]):
	            beta[i,:] -= h
	            _,_, _, _, _, c1 = self.forward_pass(X, Y)
	            beta[i, :] += 2*h
	            _,_, _, _, _, c2 = self.forward_pass(X, Y)
	            gradBeta[i, :] = (c2-c1) / (2*h)
	            beta[i, :] -= h

	        grad_gamma.append(gradGamma)
	        grad_beta.append(gradBeta)
	    return grad_W, grad_b, grad_gamma[0:-1],grad_beta[0:-1]


#3-LAYER AND 9-LAYER WITH AND WITHOUT BN
"""
a=k_layer_network(3072,10,[50,50],'he',0.005)
num_dim = 3072
num_samples = 45000
batch_size = 100
no_cycles = 2
epochs = 2*no_cycles*5*num_samples//batch_size
eta_min = 1e-5
eta_max = 1e-1
n_s = 5*450
batch_norm = True
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)


x_axis = np.linspace(0,epochs,len(train_loss))
plt.plot(x_axis,train_loss,x_axis,val_loss)
plt.xlabel('update step')
plt.ylabel('cost')
plt.legend(['training','validation'])
plt.savefig('loss_3layer_batch.png')
plt.clf()
plt.plot(x_axis,train_acc,x_axis,val_acc)
plt.xlabel('update step')
plt.ylabel('accuracy')
plt.legend(['training','validation'])
plt.savefig('accuracy_3layer_batch.png')

"""


#FINAL TEST ACCURACY OPTIMAL LAMBDA
"""
a=k_layer_network(3072,10,[50,50],'he',0.0013)
num_dim = 3072
num_samples = 45000
batch_size = 100
no_cycles = 3
epochs = 2*no_cycles*5*num_samples//batch_size
eta_min = 1e-5
eta_max = 1e-1
n_s = 5*450
batch_norm = True
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
print(final_test)
"""


#SENSITIVITY TO INITIALIZATION
"""
num_dim = 3072
num_samples = 45000
batch_size = 100
no_cycles = 3
epochs = 2*no_cycles*5*num_samples//batch_size
eta_min = 1e-5
eta_max = 1e-1
n_s = 5*450

batch_norm = True
sigma = 1e-1
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-1_BN_TESTACC.npy',final_test)
np.save('sigma1e-1_BN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-1_BN_VALLOSS.npy',val_loss)

sigma = 1e-1
batch_norm = False
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-1_noBN_TESTACC.npy',final_test)
np.save('sigma1e-1_noBN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-1_noBN_VALLOSS.npy',val_loss)

sigma = 1e-3
batch_norm = True
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-3_BN_TESTACC.npy',final_test)
np.save('sigma1e-3_BN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-3_BN_VALLOSS.npy',val_loss)

sigma = 1e-3
batch_norm = False
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-3_noBN_TESTACC.npy',final_test)
np.save('sigma1e-3_noBN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-3_noBN_VALLOSS.npy',val_loss)

sigma = 1e-4
batch_norm = True
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-4_BN_TESTACC.npy',final_test)
np.save('sigma1e-4_BN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-4_BN_VALLOSS.npy',val_loss)

sigma = 1e-4
batch_norm = False
a=k_layer_network(3072,10,[50,50],'normal',0.005,sigma)
train_loss,train_acc,val_loss,val_acc,final_test = a.train(num_dim,num_samples,batch_size,epochs,eta_min,eta_max,n_s,batch_norm)
np.save('sigma1e-4_noBN_TESTACC.npy',final_test)
np.save('sigma1e-4_noBN_TRAINLOSS.npy',train_loss)
np.save('sigma1e-4_noBN_VALLOSS.npy',val_loss)


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
"""
