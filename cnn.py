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

		deconv_layer1_weights = np.random.normal(mean, std,(filter_size, filter_size, input_channels, no_of_filters1))
		deconv_layer1_biases = np.zeros((1, no_of_filters1))
		deconv_layer2_weights = np.random.normal(mean, std,(filter_size, filter_size, no_of_filters1, no_of_filters2))
		deconv_layer2_biases = np.zeros((1, no_of_filters2))


	def sigmoid(x):
		return 1/(1+np.exp(-1*x))

	def sigmoid_derv(x):
		return sigmoid(x)*(1-sigmoid(x))


	def conv(X,w,b):
		return np.sum(np.multiply(w,X))+b

	def conv_layer(image, filter, bias):
		m,row,col,channels = image.shape
		nf = filter.shape[3]
		new_image = np.zeros((m, row-filter_size+1, col-filter_size+1, nf))

		for i in range(m):
			img = image[i]
			for r in range(row-filter_size+1):
				for c in range(col-filter_size+1):
					x = img[r:r+filter_size, c:c+filter_size, :]
					for f in range(nf):
						new_image[i,r,c,f] = conv(x,filter[:,:,:,f],bias[:,f])
		return new_image


	def maxPool(self,image):
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


    def deconv_layer(image,filter,bias):
    	m,row,col,channels = image.shape
    	pad = filter_size -1 
    	image_pad = np.zeros((m, row+2*pad, col+2*pad, channels))
		image_pad[:,pad+1:row+pad+1,pad+1:col+pad+1,:] = image
		new_image = conv_layer(image_pad, filter,bias)
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


	def forward_prop(image):
		initialize_parameters()
		conv_layer1 = conv_layer(image, conv_layer1_weights, conv_layer1_biases)
		maxpool1 = maxpool(conv_layer1)
		conv_layer2 = conv_layer(maxpool1, conv_layer2_weights, conv_layer2_biases)
		maxpool2 = maxpool(conv_layer2)
		deconv_layer1 = deconv_layer(maxpool2, deconv_layer1_weights, deconv_layer1_biases)
		unmaxpool1 = unmaxpool(deconv_layer1)
		deconv_layer2 = deconv_layer(maxpool1, deconv_layer2_weights, deconv_layer2_biases)
		unmaxpool2 = unmaxpool(deconv_layer2)

