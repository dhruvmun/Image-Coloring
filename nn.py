import numpy as np
from numpy import linalg as la

class NN:
	def __init__(self,dim):
		self.input_size = dim*dim
		self.output_size = dim*dim*3
		self.hidden_layers = [128,512]
		self.parameters = dict()
		self.derivatives = dict()
		self.layers = dict()


	def initialize_parameters(self):
		self.parameters['W1'] = np.random.randn(self.hidden_layers[0], self.input_size) * 0.01
		self.parameters['b1'] = np.zeros((self.hidden_layers[0], 1))
		self.parameters['W2'] = np.random.randn(self.hidden_layers[1], self.hidden_layers[0]) * 0.01
		self.parameters['b2'] = np.zeros((self.hidden_layers[1], 1))
		self.parameters['W3'] = np.random.randn(self.output_size, self.hidden_layers[1]) * 0.01
		self.parameters['b3'] = np.zeros((self.output_size, 1))


	def sigmoid(self,x):
		return float(1)/(1+np.exp(-1*x))

	def sigmoid_derv(self,x):
		return self.sigmoid(x)*(1-self.sigmoid(x))


	def linear_forward(self,W,X,b):
		return self.sigmoid(np.dot(W,X)+b)

	def forward_prop(self,input_image):
		self.layers['hidden_layer1'] = self.linear_forward(self.parameters['W1'], input_image, self.parameters['b1'])
		self.layers['hidden_layer2'] = self.linear_forward(self.parameters['W2'], self.layers['hidden_layer1'], self.parameters['b2'])
		self.layers['output_layer'] = self.linear_forward(self.parameters['W3'], self.layers['hidden_layer2'], self.parameters['b3'])
		#return self.layers['output_layer']

	def backward_prop(self, input_image,y):
		
		_,m = input_image.shape

		derv_hidden_layer3 = self.layers['output_layer']-y
		#derv_hidden_layer3 *= self.sigmoid_derv(self.layers['hidden_layers3'])
		self.derivatives['W3'] = np.dot(derv_hidden_layer3, self.layers['hidden_layer2'].T) /m
		self.derivatives['b3'] = np.squeeze(np.sum(derv_hidden_layer3, axis=1, keepdims=True)) / m
		
		derv_hidden_layer2 = np.dot(self.parameters['W3'].T, derv_hidden_layer3)
		derv_hidden_layer2 *= self.sigmoid_derv(self.layers['hidden_layer2'])
		self.derivatives['W2'] = np.dot(derv_hidden_layer2, self.layers['hidden_layer1'].T) /m
		self.derivatives['b2'] = np.squeeze(np.sum(derv_hidden_layer2, axis=1, keepdims=True)) / m
		
		derv_hidden_layer1 = np.dot(self.parameters['W2'].T, derv_hidden_layer2)
		derv_hidden_layer1 *= self.sigmoid_derv(self.layers['hidden_layer1'])
		self.derivatives['W1'] = np.dot(derv_hidden_layer1, input_image.T) /m
		self.derivatives['b1'] = np.squeeze(np.sum(derv_hidden_layer1, axis=1, keepdims=True)) / m


	def Loss(self,y):
		return np.mean((self.layers['output_layer']-y)**2)
		

	def learning_algorithm(self, lr):
		self.parameters['W1'] -= lr * self.derivatives['W1']
		self.parameters['b1'] -= lr * self.derivatives['b1'].reshape(self.hidden_layers[0],1)
		self.parameters['W2'] -= lr * self.derivatives['W2']
		self.parameters['b2'] -= lr * self.derivatives['b2'].reshape(self.hidden_layers[1],1)
		self.parameters['W3'] -= lr * self.derivatives['W3']
		self.parameters['b3'] -= lr * self.derivatives['b3'].reshape(self.output_size,1)
