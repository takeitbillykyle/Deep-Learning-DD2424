import numpy as np
import copy
import matplotlib.pyplot as plt

#np.random.seed(100)


class RNN():
	def __init__(self, m, eta, seq_length, params, sig=1e-2):
		self.m = m
		self.eta = eta
		self.seq_length = seq_length
		self.U = params['U']
		self.W = params['W']
		self.V = params['V']
		self.b = params['b']
		self.c = params['c']
		self.load_data()
		self.char_to_ind = params['char_to_ind']
		self.ind_to_char = params['ind_to_char']
		self.K = len(self.char_to_ind)
		self.adam_sums = {"U":np.zeros(self.U.shape),"W":np.zeros(self.W.shape),"V":np.zeros(self.V.shape),"b":np.zeros(self.b.shape),"c":np.zeros(self.c.shape)}
		self.adam_sq_sums = {"U":0,"W":0,"V":0,"b":0,"c":0}
		self.parameters = {"U": self.U, "W": self.W,
		    "V": self.V, "b": self.b, "c": self.c}
		self.beta1 = 0.9 
		self.beta2 = 0.999

	def load_data(self):
		file = open('goblet_book.txt')
		self.book_data = file.read()
	def SoftMax(self, vec):
		return np.exp(vec - np.max(vec)) / np.sum(np.exp(vec - np.max(vec)), axis=0)

	def synthesize(self, h_0, x_0, n):
		h = h_0
		x = x_0
		Y = np.zeros((self.K, n))
		for t in range(n):
			a = self.W@h+self.U@x+self.b
			h = np.tanh(a)
			o = self.V@h+self.c
			p = self.SoftMax(o)
			ind = np.where((np.cumsum(p)-np.random.uniform(0, 1)) > 0)[0][0]
			Y[ind, t] = 1
			x = Y[:,t].reshape(self.K,1)
		return Y

	def one_hot_encoding(self, vec_char):
		one_hot = np.zeros((self.K, len(vec_char)))
		for i, char in enumerate(vec_char):
			one_hot[self.char_to_ind[char], i] = 1
		return one_hot

	def Y_to_text(self, Y):
		text = ''
		for i in range(Y.shape[1]):
			ind = np.where(Y[:,i]==1)[0][0]
			text += self.ind_to_char[ind]
		return text

	def forward_pass(self, x_seq, y_seq, h_0):
		a_seq = []
		h_seq = [h_0]
		p_seq = []
		loss = 0
		tau = x_seq.shape[1]
		for t in range(tau):
			x = x_seq[:, t].reshape(x_seq.shape[0], 1)

			a = self.W@h_seq[t]+self.U@x+self.b
			a_seq.append(a)

			h = np.tanh(a)
			h_seq.append(h)

			o = self.V@h+self.c

			p = self.SoftMax(o)
			p_seq.append(p)

			loss -= np.log(y_seq[:, t].T@p)
		a_seq, h_seq, p_seq = np.array(a_seq).reshape(tau, self.m), np.array(
		    h_seq).reshape(tau+1, self.m), np.array(p_seq).reshape(tau, self.K)
		return loss, a_seq, h_seq, p_seq

	def backward_pass(self, x_seq, p_seq, y_seq, a_seq, h_seq):
		tau = len(a_seq)
		dLdo = p_seq-y_seq.T

		dLdh = np.zeros((tau, self.m))
		dLda = np.zeros((tau, self.m))

		dLdW = np.zeros((self.W.shape))
		dLdU = np.zeros((self.U.shape))
		dLdV = np.zeros((self.V.shape))
		dLdh[tau-1, :] = dLdo[-1, :]@self.V
		dLda[tau-1, :] = dLdh[tau-1, :]@np.diag(1 - np.tanh(a_seq[-1])**2)

		for t in range(tau-2, -1, -1):
			dLdh[t, :] = dLdo[t, :]@self.V + dLda[t+1, :]@self.W
			dLda[t, :] = dLdh[t, :]@np.diag(1 - np.tanh(a_seq[t]**2))
		dLdW = dLda.T@h_seq[:-1,:]
		dLdU = dLda.T@x_seq.T
		dLdV = dLdo.T@h_seq[1:,:]
		dLdb = np.sum(dLda, axis=0).reshape(self.m, 1)
		dLdc = np.sum(dLdo, axis=0).reshape(self.K, 1)
		grads = {"W": dLdW, "V": dLdV, "U": dLdU, "b": dLdb, "c": dLdc}
		for grad in grads:
			grads[grad] = np.maximum(np.minimum(grads[grad],5),-5)
		return grads

	def ada_grad(self, iter):
		h_0 = np.zeros((self.m,1))
		h=h_0
		e = 0
		losses = []
		for i in range(iter):
			if (e+self.seq_length+1)>=len(self.book_data):
				e = 0
				h = h_0
			x_seq = self.one_hot_encoding(self.book_data[e:e+self.seq_length])
			y_seq = self.one_hot_encoding(self.book_data[e+1:e+self.seq_length+1])
			loss, a_seq, h_seq, p_seq = self.forward_pass(x_seq, y_seq, h)
			grads = self.backward_pass(x_seq, p_seq, y_seq, a_seq, h_seq)
			if i == 0:
				smooth_loss = loss
			else:
				smooth_loss = 0.999*smooth_loss + 0.001*loss

			e += self.seq_length
			for k in grads:
				self.adam_sq_sums[k] = self.beta2*self.adam_sq_sums[k]+(1-self.beta2)*grads[k]**2
				self.adam_sums[k] = self.beta1*self.adam_sums[k]+(1-self.beta2)*grads[k]
				m_hat = self.adam_sums[k]/(1-self.beta1**(i+1))
				v_hat = self.adam_sq_sums[k]/(1-self.beta2**(i+1))
				self.parameters[k] -= self.eta/np.sqrt(v_hat+1e-8)*m_hat
			if i%10000 ==0:
				Y_synth = self.synthesize(h, x_seq[:,0].reshape(self.K,1), 100)
				print("---- LOSS AT ITER %.1f: " %i +str(smooth_loss) +" -----")
				print(self.Y_to_text(Y_synth))
				losses.append(float(smooth_loss))
			h = h_seq[-1].reshape(self.m,1) 
		return losses

	def compute_grads_num(self, h):
		h_0 = np.zeros((self.m, 1))
		x_seq = self.one_hot_encoding(a.book_data[0:self.seq_length])
		y_seq = self.one_hot_encoding(a.book_data[1:self.seq_length+1])
		grads = {'U': np.zeros(self.parameters['U'].shape), 'V': np.zeros(self.parameters['V'].shape),'W': np.zeros(self.parameters['W'].shape), 'b': np.zeros(self.parameters['b'].shape),'c': np.zeros(self.parameters['c'].shape)}
		loss, a_seq, h_seq, p_seq = self.forward_pass(x_seq, y_seq, h_0)
		grads_anal = self.backward_pass(x_seq, p_seq, y_seq, a_seq, h_seq)

		for k in self.parameters:
			for i in range(self.parameters[k].shape[0]):
				if self.parameters[k].ndim == 1:
					self.parameters[k][i] -= h
					l1, _,_,_ = self.forward_pass(x_seq, y_seq, h_0)
					self.parameters[k][i] += 2*h
					l2,_,_,_ = self.forward_pass(x_seq,y_seq,h_0)
					grads[k][i] = (l2-l1)/(2*h)
					self.parameters[k][i] -= h
				else:
					for j in range(self.parameters[k].shape[1]):
						self.parameters[k][i,j] -= h
						l1,_,_,_ = self.forward_pass(x_seq, y_seq, h_0)
						self.parameters[k][i,j] += 2*h
						l2, _,_,_ = self.forward_pass(x_seq,y_seq,h_0)
						grads[k][i,j] = (l2-l1)/(2*h)
						self.parameters[k][i,j] -= h
		print(self.compute_relative_error(grads['V'],grads_anal['V'],1e-10))
		print(self.acceptable_ratio(self.compute_relative_error(grads['W'],grads_anal['W'],1e-10),1e-6))
	def acceptable_ratio(self,error,tolerance):
		return 100*np.sum(np.array(error)<tolerance)/np.size(error)

		return grads
	def compute_relative_error(self,grad_num, grad_anal, eps):
		error = np.abs(grad_num - grad_anal)/np.maximum(eps, np.abs(grad_num) + np.abs(grad_anal))
		return error

	def write_text(self,seed,n):
		h_0 = np.zeros((self.m, 1))
		x_seq = self.one_hot_encoding(seed[:-1])
		y_seq = self.one_hot_encoding(seed[1:])
		loss, a_seq, h_seq, p_seq=self.forward_pass(x_seq, y_seq, h_0)
		Y=self.synthesize(h_seq[-1,:].reshape(self.m,1), x_seq[:,-1].reshape(self.K,1), n)
		text = seed+self.Y_to_text(Y)
		return text

#a = RNN(5,0.01,25)
#a.compute_grads_num(1e-4)
params = np.load('best_params.npy').item()
params['char_to_ind'] = np.load('char_to_ind.npy').item()
params['ind_to_char'] = np.load('ind_to_char.npy').item()
a = RNN(100,0.007235714285714286,25,params)
print(a.write_text("Harr", 996))

#np.save('losses_adam.npy',losses)
