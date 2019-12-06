import numpy as np
import cnn
from numpy import linalg as la
class Pooling:
    def __init__(self,size):
        poolSize=size


    def maxPool(self,image):
        count_images,row,col,channels=image.shape
        new_row=row//2
        new_col=col//2
        new_image=np.zeros([count_images,new_row,new_col,channels])

        for i in range(count_images):
            img=image[i]
            for r in range(new_row):
                for c in range(new_col):
                    for d in range(channels):
                         new_image[i,r,c,d]=findMax(img[r:2*r+2,c:2*c+2,d])
        return new_image


    def findMax(self,image):
        return np.max(image)

    def unPooling(self,image):
        count_image,row,col,channels=image.shape
        new_image=np.zeros([count_image,2*row,2*col,channels])
        for i in range(count_image):
            img=image[i]
            for r in range(row):
                for c in range(col):
                    for d in range(channels):
                        new_image[i,r:r+2,c:c+2,d]=img[r,c,d]

        return new_image

    def Loss(self,input,output,order):
        count,row,col,channel=input.shape
        # no_of_images,row_images,col_images,channel_images=output.shape
        loss = np.zeros([count,row, col])
        for i in range(count):
            img=input[i]
            img2=output[i]
            loss1=loss[i]
            for r in range(row):
                for c in range(col):
                #     for d in range(channel):
                    a=img[r,c]
                    b=img2[r,c]
                    loss1[r,c]=la.norm((a - b), ord=order)

        return (np.sum(loss))/row*col*channel
