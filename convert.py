import cv2
import pickle as pkl
import os
import numpy as np

def convertBW(path):
	im = cv2.imread(path+'.jpg')
	grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(path+'B'+'.jpg',grayIm)

	# return grayIm

for i in range(10):
	convertBW('dataset/beach'+str(i))

def extractImages(path, file):
	f = open(path+file, 'rb')
	dataset = pkl.load(f)
	os.mkdir(file+"B")
	os.mkdir(file+"C")
	# categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	# for category in categories:
	# 	os.mkdir(file+"B/"+category+"B")
	# 	os.mkdir(file+"C/"+category+"C")
	count = np.zeros(10,dtype=int)
	for i in range(len(dataset['data'])):
		img = dataset['data'][i]
		label = dataset['labels'][i]
		image = img.reshape((3,32,32)).transpose(1,2,0)
		imageC = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		imageB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(file+"B/"+categories[label]+"B/B"+str(count[label])+".png",imageB)
		cv2.imwrite(file+"C/"+categories[label]+"C/C"+str(count[label])+".png",imageC)
		count[label] += 1

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pkl.load(f)
    return dict

# categories = extractCategories("../cifar-10-batches-py/", "batches.meta")

# extractImages("./","data_batch_1")
# extractImages("./","data_batch_2")
# extractImages("./","data_batch_3")
# extractImages("./","data_batch_4")
# extractImages("./","data_batch_5")
