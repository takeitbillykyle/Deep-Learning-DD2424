import numpy as np
import matplotlib.pyplot as plt
import copy
#from keras.preprocessing.image import ImageDataGenerator
import imgaug as ia
import imgaug.augmenters as iaa

datapath = 'C:\\Users\\arvid\\Documents\\KTH\\Masterkurser\\Deep Learning\\Assignments\\Assignment 2\\'



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encoding(labels):
	t = np.concatenate((np.arange(len(labels)).reshape(len(labels),1),np.array(labels).reshape(len(labels),1)),axis=1)
	one_hot = np.zeros((len(labels),10))
	one_hot[t[:,0],t[:,1]] = 1
	return one_hot.T

def load_batch(file):
	data_as_dict = unpickle(file)
	img_data = data_as_dict[b'data']
	labels = data_as_dict[b'labels']
	labels_encoded = one_hot_encoding(labels)
	return img_data, labels, labels_encoded

def preprocess(tr_data,val_data,test_data):
	tr_mean = np.mean(tr_data,axis=0)
	tr_std = np.std(tr_data,axis=0)

	tr_data = (tr_data - tr_mean)/tr_std
	val_data = (val_data - tr_mean)/tr_std
	test_data = (test_data - tr_mean)/tr_std

	return tr_data.T, val_data.T, test_data.T


X_train1, y_train1, Y_train1 = load_batch(datapath+'data_batch_1')
X_train2, y_train2, Y_train2 = load_batch(datapath+'data_batch_2')
X_train3, y_train3, Y_train3 = load_batch(datapath+'data_batch_3')
X_train4, y_train4, Y_train4 = load_batch(datapath+'data_batch_4')
X_trainval, y_trainval, Y_trainval= load_batch(datapath+'data_batch_5')
X_train,y_train,Y_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_trainval[0:len(X_trainval)//2])), np.concatenate((y_train1,y_train2,y_train3,y_train4,y_trainval[0:len(y_trainval)//2])),np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_trainval[:,0:Y_trainval.shape[1]//2]),axis=1)
X_val,y_val,Y_val = X_trainval[X_trainval.shape[0]//2:], y_trainval[len(y_trainval)//2:],  Y_trainval[:,Y_trainval.shape[1]//2:]
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_test,y_test,Y_test = load_batch(datapath+'test_batch')
X_train,X_val,X_test=preprocess(X_train,X_val,X_test)

pic = X_test[:,4].reshape(3,32,32).transpose(1,2,0)
print(pic)
seq = iaa.Sequential([
	#iaa.Fliplr(0.5),
	iaa.Rotate((-45,45),mode=ia.ALL),
	#iaa.AdditiveGaussianNoise(scale=0.1*255)
	#iaa.TranslateX(percent=(-0.5,0.5))
	])

for i in range(10):
	plt.subplot(2,5,i+1)
	img = seq(image=pic)
	plt.imshow(img)
plt.show()


