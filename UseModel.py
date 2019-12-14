import cv2
import pickle as pkl
import os
import numpy as np
import nn
from PIL import Image
import pdb

def sigmoid(x):
	return float(1)/(1+np.exp(-1*x))

def linear_forward(W,X,b):
	return sigmoid(np.dot(W,X)+b)

def linear_forward_without_sigmoid(self,W,X,b):	
	return np.dot(W,X)+b


def forward_prop(input_image,parameters):
	layers = {}
	layers['hidden_layer1'] = linear_forward(parameters['W1'], input_image, parameters['b1'])
	layers['hidden_layer2'] = linear_forward(parameters['W2'], layers['hidden_layer1'], parameters['b2'])
	layers['output_layer'] = linear_forward_without_sigmoid(parameters['W3'], layers['hidden_layer2'], parameters['b3'])
	return layers['output_layer']

def giveResult(input_path, output_path, parameters):
	input_image = Image.open(input_path)
	input_image = np.asarray(input_image, dtype=np.float128).flatten()
	input_image = input_image.reshape(1024,1)
	input_image /= float(255)
	output_image = forward_prop(input_image, parameters)
	output_image *= 255
	output_image = output_image.astype(int)
	# pdb.set_trace()
	output_image = output_image.reshape(32,32,3)
	cv2.imwrite(output_path,output_image)


modelPath = '1000Model.pkl'
index = 6
# input_path = './cifar-10-batches-py/data_batch_2B/deerB/B'+str(index)+'.png'
input_path = 'B'+str(index)+'.png'
parameters = pkl.load(open(modelPath,'rb'))
giveResult(input_path, "output.png", parameters)


