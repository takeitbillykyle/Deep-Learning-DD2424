import numpy as np

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

a=k_layer_network(3,5,[1,2],'he',0)