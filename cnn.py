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

	def initialize_derivatives(self, m):
		self.derivatives['conv_layer1_weights'] = np.zeros((self.filter_size, self.filter_size, self.input_channels, self.no_of_filters1))
		self.derivatives['conv_layer1_biases'] = np.zeros((1, self.no_of_filters1))
		self.derivatives['conv_layer2_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters1, self.no_of_filters2))
		self.derivatives['conv_layer2_biases'] = np.zeros((1, self.no_of_filters2))

		self.derivatives['deconv_layer1_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters2, self.no_of_filters1))
		self.derivatives['deconv_layer1_biases'] = np.zeros((1, self.no_of_filters2))
		self.derivatives['deconv_layer2_weights'] = np.zeros((self.filter_size, self.filter_size, self.no_of_filters1, self.output_channels))
		self.derivatives['deconv_layer2_biases'] = np.zeros((1, self.no_of_filters1))

	

	def sigmoid(self, x):
		return 1/(1+np.exp(-1*x))

	def sigmoid_derv(self, x):
		return x*(1-x)


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


	def conv_backprop(self,dz,z,filter,bias,image):
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
						da[r:r+self.filter_size, c:c+self.filter_size, :] += sigmoid_derv(z[i,r,c,f])*filter[:,:,:,f]*dz[i,r,c,f]
						dW[:,:,:,f] += sigmoid_derv(z[i,r,c,f])*x * dz[i, r, c, f]
						db[:,:,:,f] += sigmoid_derv(z[i,r,c,f])*dz[i, r, c, f]
			derv_prev_image[i, :, :, :] = da[pad:-pad, pad:-pad, :]
		return derv_prev_image


	def maxpool_backprop(self,der_deconv,layers):
		# count, row, col, channels = der_deconv.shape
		prev_count,prev_row,prev_col,prev_channels=layers.shape
		conv_layer=np.zeros([prev_count,prev_row,prev_col,prev_channels])
		for i in range(prev_count):
			img=der_deconv[i]
			img2=layers[i]
			for d in range(prev_channels):
				for j in range(0,prev_row,2):
					for k in range(0,prev_col,2):
						arr=img2[j,k,d]
						result=np.where(arr==np.amax(arr))
						coordinates=list(zip(result[0],result[1],result[2]))
						for cord in coordinates:
							x,y,z=cord
							img2[x,y,z]=img[j//2,k//2,d]
			conv_layer[i]=img2
		return conv_layer



	def unpool_backprop(self,derv_deconv):
		count,row,col,channels=derv_deconv.shape
		unpooled_image=np.zeros([count,row//2,col//2,channels])
		for i in range(count):
			img=derv_deconv[i]
			unpool=unpooled_image[i]
			for d in range(channels):
				for j in range(0,row,2):
					for k in range(0,col,2):
						unpool[j//2,k//2,d]=np.sum(img[j:j+2,k:k+2,d])

		return unpooled_image

	#def learning_algorithm():



	def backprop(self,loss):

		derv_deconv2 = self.conv_backprop(loss,self.derivatives['deconv_layer2_weights'],self.derivatives['deconv_layer2_biases'],self.layers['unpool2'])
		print(derv_deconv2.shape)
		derv_unpool2 = self.unpool_backprop(derv_deconv2)
		print(derv_unpool2.shape)
		derv_deconv1 = self.conv_backprop(derv_unpool2,self.derivatives['deconv_layer1_weights'],self.derivatives['deconv_layer1_biases'],self.layers['unpool1'])
		print(derv_deconv1.shape)
		derv_unpool1 = self.unpool_backprop(derv_deconv2)
		print(derv_unpool2.shape)		
		derv_pool2 = self.maxpool_backprop(derv_unpool1, self.layers['conv_layer2'])
		print(derv_pool2.shape)
		derv_conv2 = self.conv_backprop(derv_pool2,self.derivatives['conv_layer2_weights'],self.derivatives['conv_layer2_biases'],self.layers['maxpool1'])
		print(derv_conv2.shape)
		derv_pool1 = self.maxpool_backprop(derv_conv2, self.layers['conv_layer1'])
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
			for d in range(channel):
				for r in range(row):
					for c in range(col):
						a = img[r, c, d]
						b = img2[r, c, d]
						loss1[r, c,d] = la.norm((a-b), ord=order)

		return (np.sum(loss)) / (row * col * channel * count)


	def train_model(self,inputs,outputs,batch_size,iters):
		self.initialize_parameters()
		self.initialize_derivatives()
		J=[]


		for step in range(iters):
			start = (step*batch_size)%(inputs.shape[0])
			end = start+batch_size

			batch_input = inputs[start:end,:,:,:]
			batch_output = outputs[start:end,:]

			self.forward_prop(batch_input)
			loss = self.Loss(batch_output, self.layers['deconv_layer2'], order)
			self.backprop(loss)
			self.learning_algorithm()

			#append the loss and accuracy of every batch
			J.append(loss)

			#print loss and accuracy of the batch dataset.
			if(step%100==0):
				print('Step : %d'%step)
				print('Loss : %f'%loss)

		return J,A



