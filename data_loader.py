from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
    
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = 'data/' + label_filename + '.zip'
        image_zip = 'data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            images = images
        return images, labels

    # Get pair of (image,label) of the current minibatch/chunk
    def create_batches(self,x,y,batch_size):
        mini_x = [x[i:i+batch_size] for i in xrange(0,60000,batch_size)]
        mini_y = [y[i:i+batch_size] for i in xrange(0,60000,batch_size)]
        return mini_x,mini_y

     



