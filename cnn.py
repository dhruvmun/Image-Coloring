import numpy as np

class CNN:
	def __init__(self):
		self.image_size = 28 
		self.input_channels = 1
		self.output_channels = 3
		self.filter_size = 3
		self.no_of_filters1 = 32
		self.no_of_filters2 = 64
		self.learning_rate = 0.1
		self.parameters = {}

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
						new_image[i,r,c,f] = self.conv(x,filter[:,:,:,f],bias[:,f])
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
		conv_layer1 = self.conv_layer(image, self.parameters['conv_layer1_weights'], self.parameters['conv_layer1_biases'], True)
		print(conv_layer1.shape)
		maxpool1 = self.maxpool(conv_layer1)
		print(maxpool1.shape)
		conv_layer2 = self.conv_layer(maxpool1, self.parameters['conv_layer2_weights'], self.parameters['conv_layer2_biases'], True)
		print(conv_layer2.shape)
		maxpool2 = self.maxpool(conv_layer2)
		print(maxpool2.shape)
		unpool1 = self.unPooling(maxpool2)
		print(unpool1.shape)
		deconv_layer1 = self.deconv_layer(unpool1, self.parameters['deconv_layer1_weights'], self.parameters['deconv_layer1_biases'])
		print(deconv_layer1.shape)
		unmaxpool2 = self.unPooling(deconv_layer1)
		print(unmaxpool2.shape)
		deconv_layer2 = self.deconv_layer(unmaxpool2, self.parameters['deconv_layer2_weights'], self.parameters['deconv_layer2_biases'])
		print(deconv_layer2.shape)
		return deconv_layer2

#image = np.random.randn(1,32,32,1)
#cnn = CNN()
#cnn.initialize_parameters()
#new_image = cnn.forward_prop(image)
#print(new_image.shape)