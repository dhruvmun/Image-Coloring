import numpy as np

class CNN:
	def __init__:
		image_size = 28 
		input_channels = 1
		output_channels = 3
		filter_size = 3
		no_of_filters1 = 32
		no_of_filters2 = 64
		learning_rate = 0.1

	def initialize_parameters():
		mean = 0
		std = 0.5

		conv_layer1_weights = np.random.normal(mean, std,(filter_size, filter_size, input_channels, no_of_filters1))
    	conv_layer1_biases = np.zeros((1, no_of_filters1))
    	conv_layer2_weights = np.random.normal(mean, std,(filter_size, filter_size, no_of_filters1, no_of_filters2))
    	conv_layer2_biases = np.zeros((1, no_of_filters2))


	def sigmoid(x):
		return 1/(1+np.exp(-1*x))

	def sigmoid_derv(x):
		return sigmoid(x)*(1-sigmoid(x))


	def conv(X,w,b):
		return np.sum(np.multiply(w,X))+b

	def conv_layer(image, filter, bias):
		m,row,col,channels = image.shape
		nf = filter.shape[3]
		new_image = np.zeros((m, n-filter_size+1, n-filter_size+1, nf))

		for i in m:
			img = image[i]
			for r in range(row-filte_size):
				for c in range(col-filter_size):
					x = img[r:r+filter_size][c:c+filter_size][:]
					new_image = conv(x,filter,bias[])






