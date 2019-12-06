import numpy as np
from numpy import linalg as la

class CNN:
	def __init__(self):
		self.image_size = 28
		self.input_channels = 1
		self.output_channels = 3
		self.filter_size = 3
		self.no_of_filters1 = 32
		self.no_of_filters2 = 64
		self.learning_rate = 0.1
		self.parameters = dict()
		self.derivatives = dict()
		self.layers = dict()


	def initialize_parameters(self):
		mean = 0
		std = 0.5

		self.parameters['conv_layer1_weights'] = np.random.normal(mean, std,(self.filter_size, self.filter_size, self.input_channels, self.no_of_filters1))
		self.parameters['conv_layer1_biases'] = np.zeros((1, self.no_of_filters1))
		self.parameters['conv_layer2_weights'] = np.random.normal(mean, std,(self.filter_size, self.filter_size, self.no_of_filters1, self.no_of_filters2))
		self.parameters['conv_layer2_biases'] = np.zeros((1, self.no_of_filters2))	

		self.parameters['deconv_layer1_weights'] = np.random.normal(mean, std,(self.filter_size, self.filter_size, self.no_of_filters2, self.no_of_filters1))
		self.parameters['deconv_layer1_biases'] = np.zeros((1, self.no_of_filters2))
		self.parameters['deconv_layer2_weights'] = np.random.normal(mean, std,(self.filter_size, self.filter_size, self.no_of_filters1, self.output_channels))
		self.parameters['deconv_layer2_biases'] = np.zeros((1, self.no_of_filters1))


	def sigmoid(self, x):
		return 1/(1+np.exp(-1*x))

	def sigmoid_derv(self, x):
		return self.sigmoid(x)*(1-self.sigmoid(x))


	def conv(self, X,w,b):
		return np.sum(np.multiply(w,X))+b

	def conv_layer(self, image, filter, bias, toPad):
		if toPad == True:
			pad = (self.filter_size -1)//2
			image = np.pad(image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
		
		m,row,col,channels = image.shape
		nf = filter.shape[3]
		new_image = np.zeros((m, row-self.filter_size+1, col-self.filter_size+1, nf))

		for i in range(m):
			img = image[i]
			for r in range(row-self.filter_size+1):
				for c in range(col-self.filter_size+1):
					x = img[r:r+self.filter_size, c:c+self.filter_size, :]
					for f in range(nf):
						new_image[i,r,c,f] = self.sigmoid(self.conv(x,filter[:,:,:,f],bias[:,f]))
		return new_image


	def maxpool(self,image):
		count_images,row,col,channels = image.shape
		new_row=row//2
		new_col=col//2
		new_image=np.zeros((count_images,new_row,new_col,channels))

		for i in range(count_images):
		    img=image[i]
		    for r in range(new_row):
		        for c in range(new_col):
		            for d in range(channels):
		                 new_image[i,r,c,d]=np.max(img[2*r:2*r+2,2*c:2*c+2,d])
		return new_image


	def deconv_layer(self,image,filter,bias):
		pad = (self.filter_size -1)//2
		image_pad = np.pad(image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
		new_image = self.conv_layer(image_pad, filter, bias, False)
		return new_image


	def unPooling(self,image):
		count_image,row,col,channels=image.shape
		new_image=np.zeros((count_image,2*row,2*col,channels))
		for i in range(count_image):
			img=image[i]
			for r in range(row):
				for c in range(col):
					for d in range(channels):
						new_image[i,2*r:2*r+2,2*c:2*c+2,d]=img[r,c,d]
		return new_image


	def forward_prop(self,image):
		self.initialize_parameters()
		print(image.shape)
		self.layers['conv_layer1'] = self.conv_layer(image, self.parameters['conv_layer1_weights'], self.parameters['conv_layer1_biases'], True)
		print(self.layers['conv_layer1'].shape)
		self.layers['maxpool1'] = self.maxpool(self.layers['conv_layer1'])
		print(self.layers['maxpool1'].shape)
		self.layers['conv_layer2'] = self.conv_layer(self.layers['maxpool1'], self.parameters['conv_layer2_weights'], self.parameters['conv_layer2_biases'], True)
		print(self.layers['conv_layer2'].shape)
		self.layers['maxpool2'] = self.maxpool(self.layers['conv_layer2'])
		print(self.layers['maxpool2'].shape)
		self.layers['unpool1'] = self.unPooling(self.layers['maxpool2'])
		print(self.layers['unpool1'].shape)
		self.layers['deconv_layer1'] = self.deconv_layer(self.layers['unpool1'], self.parameters['deconv_layer1_weights'], self.parameters['deconv_layer1_biases'])
		print(self.layers['deconv_layer1'].shape)
		self.layers['unpool2'] = self.unPooling(self.layers['deconv_layer1'])
		print(self.layers['unpool2'].shape)
		self.layers['deconv_layer2'] = self.deconv_layer(self.layers['unpool2'], self.parameters['deconv_layer2_weights'], self.parameters['deconv_layer2_biases'])
		print(self.layers['deconv_layer2'].shape)


	def initialize_derivatives(self, m):
		self.derivatives['conv_layer1_weights'] = np.zeros((self.filter_size, self.filter_size, self.input_channels, self.no_of_filters1))
		self.derivatives['conv_layer1_biases'] = np.zeros((1, self.no_of_filters1))
		self.derivatives['conv_layer2_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters1, self.no_of_filters2))
		self.derivatives['conv_layer2_biases'] = np.zeros((1, self.no_of_filters2))

		self.derivatives['deconv_layer1_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters2, self.no_of_filters1))
		self.derivatives['deconv_layer1_biases'] = np.zeros((1, self.no_of_filters2))
		self.derivatives['deconv_layer2_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters1, self.output_channels))
		self.derivatives['deconv_layer2_biases'] = np.zeros((1, self.no_of_filters1))


	def conv_backprop(self,dz,filter,bias,image):
		m,row1,col1,channels1 = image.shape
		nf = filter.shape[3]
		_,row2,col2,_ = dz.shape

		derv_prev_image = np.zeros(image.shape)
		pad = (self.filter_size -1)//2
		dA_pad = np.pad(derv_prev_image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
		image_pad = np.pad(image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
		
		for i in range(m):
			img = image_pad[i]
			da = dA_pad[i]
			for r in range(row2): 
				for c in range(col2):
					x = img[r:r+self.filter_size, c:c+self.filter_size, :]
					for f in range(nf):
						da[r:r+self.filter_size, c:c+self.filter_size, :] += filter[:,:,:,c] * dz[i, h, w, c]
						dW[:,:,:,c] += x * dz[i, h, w, c]
						db[:,:,:,c] += dz[i, h, w, c]
			derv_prev_image[i, :, :, :] = da[pad:-pad, pad:-pad, :]
		return derv_prev_image


	#def maxpool_backprop():


	#def unpool_backprop():


	#def learning_algorithm():


	def backprop(self):
		self.initialize_derivatives()
		derv_deconv2 = self.conv_backprop(loss,self.derivatives['deconv_layer2_weights'],self.derivatives['deconv_layer2_biases'],self.layers['unpool2'])
		print(derv_deconv2.shape)
		derv_unpool2 = self.unpool_backprop(derv_deconv2, self.layers['deconv_layer1'])
		print(derv_unpool2.shape)
		derv_deconv1 = self.conv_backprop(derv_unpool2,self.derivatives['deconv_layer1_weights'],self.derivatives['deconv_layer1_biases'],self.layers['unpool1'])
		print(derv_deconv1.shape)
		derv_unpool1 = self.unpool_backprop(derv_deconv2, self.layers['maxpool2'])
		print(derv_unpool2.shape)		
		derv_pool2 = self.unpool_backprop(derv_unpool1, self.layers['conv_layer2'])
		print(derv_pool2.shape)
		derv_conv2 = self.conv_backprop(derv_pool2,self.derivatives['conv_layer2_weights'],self.derivatives['conv_layer2_biases'],self.layers['maxpool1'])
		print(derv_conv2.shape)
		derv_pool1 = self.unpool_backprop(derv_conv2, self.layers['conv_layer1'])
		print(derv_pool1.shape)
		derv_conv1 = self.conv_backprop(derv_pool1,self.derivatives['conv_layer1_weights'],self.derivatives['conv_layer1_biases'],self.layers['image'])
		print(derv_conv1.shape)

	def Loss(self, input, output, order):
		count, row, col, channel = input.shape
		# no_of_images,row_images,col_images,channel_images=output.shape
		loss = np.zeros([count, row, col,channel])
		for i in range(count):
			img = input[i]
			img2 = output[i]
			loss1 = loss[i]
			for r in range(row):
				for c in range(col):
					for d in range(channel):
						a = img[r, c, d]
						b = img2[r, c, d]
						loss1[r, c,d] = la.norm((a - b), ord=order)

		return (np.sum(loss)) / (row * col * channel * count)


image = np.random.randn(1,32,32,1)
cnn = CNN()
#cnn.initialize_parameters()
new_image = cnn.forward_prop(image)
#print(new_image.shape)


