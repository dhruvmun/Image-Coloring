import numpy as np
import cnn
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
                         new_image[i,r,c,d]=findMax(img[r:r+2,c:c+2,d])



    def findMax(self,image):
        return np.max(image)