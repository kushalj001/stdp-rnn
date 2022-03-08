from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import matlab.engine
import scipy.io as sio
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def normalize_data(data, maximum = -1):
    if maximum == -1:
        maximum = np.max(data)
        
    if np.issubclass_(data.dtype.type, np.integer):
        data = data.astype('float32')
        
    data /= maximum
    return data

def separate_classes(data, truth, **kwargs):
    data_by_class = []
    truth_by_class = []
    classes = np.unique(truth)
    n_classes = len(classes)
    df = K.image_data_format()
    
    for c in classes:
        #get the training data
        inds = np.argwhere(truth == c)
        examples = data[inds]
        
        if df == 'channels_last':
            np.squeeze(examples, axis=1)
            np.expand_dims(examples, axis=3)
            
        #this changes the shape to (#ex, width*height)
        if 'flatten' in kwargs and kwargs['flatten']:
            examples = np.squeeze(examples, axis=1)
            examples = np.reshape(examples, (-1, np.prod(examples.shape[1:])))
              
        data_by_class.append(examples)
        n_examples = examples.shape[0]
        truth_by_class.append(np.ones(n_examples, dtype='int') * c)
    

    if 'return_classes' in kwargs and kwargs['return_classes']:
        return data_by_class, truth_by_class, classes
    else:
        return data_by_class, truth_by_class

def class_accuracy(m, x_test_sep, y_test_sep):
    n_classes = len(y_test_sep)
    acc = np.zeros(n_classes)
    truth = convert_to_categorical(y_test_sep)
    for i in range(n_classes):
        acc[i] = m.evaluate(x_test_sep[i], truth[i])[1]
    	print("Accuracy on digit " + str(i) + " = " + str(acc[i]))
    return acc

def convert_to_categorical(truth_sep):
    #convert truth to categorical if necessary
    if truth_sep[0][0].shape == ():
        print("Converting truth to categorical...")
        truth_categorical = []
        n_classes = len(truth_sep)
        
        for i in range(n_classes):
            truth_categorical.append(keras.utils.to_categorical(truth_sep[i], n_classes))
            
        return truth_categorical
    else:
        return truth_sep

def concatenate_x_data(classes, x_train_sep, num_examples=5000):
	train_data = x_train_sep[classes[0]][:num_examples]
	for i in range(1,len(classes)):
		train_data = np.concatenate((train_data, x_train_sep[classes[i]][:num_examples]), axis=0)
	return train_data

def train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, *classes, **kwargs):
    available_classes = [y[0] for y in y_train_sep]
    n_classes = len(available_classes)
    
    #get params
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
    else:
        epochs = 2
        
    if 'batch size' in kwargs:
        batch_size = kwargs['batch size']
    else:
        batch_size = 100
            
    
    assert len(classes) > 0, "Must provide index(es) of class(es) to train on."
    for c in classes:
        assert c in available_classes, "Requested training index out of dataset bounds."
        
    #convert truth to categorical if necessary
    y_train_sep_categorical = convert_to_categorical(y_train_sep)
    y_test_sep_categorical = convert_to_categorical(y_test_sep)
        
    
    print("Training on classes", classes)
    #setup the arrays for data
    #add the first class
    c0 = classes[0]
    train_data = x_train_sep[c0][0:2500]
    train_truth = y_train_sep_categorical[c0][0:2500]
    test_data = x_test_sep[c0]
    test_truth = y_test_sep_categorical[c0]
    #add the rest on if training more than one class
    if len(classes) > 1:
        for i,c in enumerate(classes[1:]):
            train_data = np.concatenate((train_data, x_train_sep[c][0:2500]), axis=0)
            train_truth = np.concatenate((train_truth, y_train_sep_categorical[c][0:2500]), axis=0)
            test_data = np.concatenate((test_data, x_test_sep[c]), axis=0)
            test_truth = np.concatenate((test_truth, y_test_sep_categorical[c]), axis=0)
    
    #do the training
    model.fit(train_data, 
              train_truth,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_data, test_truth))
    score = model.evaluate(test_data, test_truth, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def sleep(numiterations, classes, model, taskname):
	# save weights and inputto mat file	
	weights = model.get_weights()

	x_data = concatenate_x_data(classes, x_train_sep, 5000)
	sio.savemat(taskname+'task_features.mat', {'w1': weights[0], 'w2':weights[1], 'w3': weights[2], 'xdata': x_data})

	# sleep
	eng = matlab.engine.start_matlab()
	eng.sleepnn_mat2py(numiterations, taskname)
	eng.quit()

	# load back weights
	weights = sio.loadmat(taskname+'task_weights.mat')
	newweights = [weights['neww1'], weights['neww2'], weights['neww3']]

	model.set_weights(newweights)
	return model

x_train = normalize_data(x_train)
x_test = normalize_data(x_test)

(x_train_sep, y_train_sep, c) = separate_classes(x_train, y_train, return_classes=True, flatten=True)
(x_test_sep, y_test_sep) = separate_classes(x_test, y_test, flatten=True)
input_shape = x_train_sep[0][0].shape
num_classes = len(c)

# Set up model architecture
model = Sequential()
#model.add(Reshape((np.prod(input_shape),), input_shape=input_shape))
model.add(Dense(1200, input_shape=(np.prod(input_shape),), activation='relu', use_bias=False))
model.add(Dropout(0.25))
model.add(Dense(1200, activation='relu', use_bias=False))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='relu', use_bias=False))

sgd = keras.optimizers.SGD(lr=0.1, momentum=0.5)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=sgd,
              metrics=['accuracy'])
class_accuracy(model, x_test_sep, y_test_sep)


train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, 0,1)
class_accuracy(model, x_test_sep, y_test_sep)

model = sleep(15000, np.arange(2), model, 'first')
class_accuracy(model, x_test_sep, y_test_sep)
train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, 2,3)
class_accuracy(model, x_test_sep, y_test_sep)
model = sleep(15000, np.arange(4), model, 'second')
class_accuracy(model, x_test_sep, y_test_sep)



train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, 4,5)
class_accuracy(model, x_test_sep, y_test_sep)
model = sleep(15000, np.arange(6), model, 'second')
class_accuracy(model, x_test_sep, y_test_sep)

train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, 6,7)
class_accuracy(model, x_test_sep, y_test_sep)
model = sleep(15000, np.arange(8), model, 'second')
class_accuracy(model, x_test_sep, y_test_sep)

train_on_set(model, x_train_sep, y_train_sep, x_test_sep, y_test_sep, 8,9)
class_accuracy(model, x_test_sep, y_test_sep)
model = sleep(15000, np.arange(10), model, 'second')
class_accuracy(model, x_test_sep, y_test_sep)
