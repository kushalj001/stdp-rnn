from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, Dropout
import numpy as np
from keras.datasets import mnist
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import tensorflow as tf
from sleepnn import *


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def compute_sequence(order, labels, X, max_seq_length):
	# order: series of digits
	# labels: image labels
	# X: images
	# max_seq_length: 15
	num_examples, inp_dimension = X.shape # number of examples and size of each example in inp dataset
	seq = np.zeros((max_seq_length, inp_dimension))
	# seq = [15, 784]
	for i in range(len(order)):
		next_class = order[i]
		indices = np.where(labels==next_class)
		image = X[indices[0][np.random.randint(0,len(indices[0]))],:]
		# select a random image with the label = next_class
		seq[i,:] = image
	return seq

def build_sequences(X, y, num_sequences, sequence_classes, max_seq_length):
	# num_sequences = 1000

	num_classes = len(sequence_classes) # number of sequence classifications
	# 10
	sequences = []
	newY = []
	for i in range(num_classes):
		for j in range(num_sequences):
			seq = compute_sequence(sequence_classes[i], y, X, max_seq_length)
			# seq = [15, 784]
			newY.append(i)
			sequences.append(seq)
		# at the end of the inner loop, sequences would be a list of [15,784] dim tensors of length 1000.
	# creating 1000 sequences of images for each sequence class.
	# 0:[4,4,6,7,1,3,6,5,8,7,6]; e.g., there will be 1000 different sequences of images
	# corresponding to this sequence of digits.
	newX = np.asarray(sequences)
	# [10000, 15, 784]
	newY = np.asarray(newY)
	return newX, newY

def build_model(hidden_size, max_seq_length, use_dropout=False):
	model = Sequential()
	model.add(SimpleRNN(hidden_size, activation='relu', input_shape=(max_seq_length,784), use_bias=False))
	# if use_dropout:
	#	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu',use_bias=False))
	model.add(Dense(10, activation='sigmoid', use_bias=False))	
	opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
	print(model.summary())
	return model

sequence_classes = {0:[4,4,6,7,1,3,6,5,8,7,6], 1:[1,3,7,3,5,6,1,9,6,1],
					2:[3,5,4,4,5,9,0,9,3,2,8,2,4,1], 3:[9,3,1,4,4,8,7,2,9,2,9,1,8], 
					4:[3,6,7,5,0,8,2,4,6,9], 5:[4,5,4,0,6,8,1,6,4,2,1,2,8,2], 
					6:[7,5,3,4,5,1,9,7,9,3,1,7,8], 7:[0,2,8,9,0], 
					8:[6,9,9,5,9], 9:[2,1,0,2,4,2,6,9,7,5,5]
					}
max_seq_length = 15

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))
X_train = X_train/255.0
X_test = X_test/255.0

seqX_train, seqY_train = build_sequences(X_train, y_train, 1000, sequence_classes, max_seq_length)
seqX_test, seqY_test = build_sequences(X_test, y_test, 100, sequence_classes, max_seq_length)

seqY_train = one_hot(seqY_train, 10)
seqY_test = one_hot(seqY_test, 10)
model = build_model(100, 1, max_seq_length)

t1 = [0,1,2,3,4,5,6,7,8,9]
t2 = [1,2,3,7,8]

labels = np.argmax(seqY_train,axis=1)
print(labels)
t1_inds = [x for x in range(len(labels)) if labels[x] in t1]
#t2_inds = [x for x in range(len(labels)) if labels[x] in t2]

testlabels = np.argmax(seqY_test,axis=1)
print(labels)
test1_inds = [x for x in range(len(testlabels)) if testlabels[x] in t1]
#test2_inds = [x for x in range(len(testlabels)) if testlabels[x] in t2]

hist1 = model.fit(seqX_train[t1_inds,:],
          seqY_train[t1_inds,:],
          epochs=2, batch_size=1000,
          validation_data=(seqX_test[test1_inds,:], seqY_test[test1_inds,:]), shuffle=True)

#hist2 = model.fit(seqX_train[t2_inds,:],
#          seqY_train[t2_inds,:],
#          epochs=10, batch_size=1000,
#          validation_data=(seqX_test, seqY_test), shuffle=True)

from matplotlib import pyplot as plt

#fig = plt.figure()
#plt.plot(np.arange(3),hist1.history['val_categorical_accuracy'])
#plt.xlabel('Training Epoch')
#plt.ylabel('Classifiaction accuracy')
#plt.title('Catastrophic forgetting on sequential learning task')
#plt.show()

params = {} # sleep params
params['inc'] 		= 0.001			# Magnitude of weight increase upon STDP event
params['dec'] 		= 0.0001 		# Magnitude of weight decrease upon STDP event
params['max_rate'] 	= 32.			# Maximum firing rate of neurons in the input layer
params['dt'] 		= 0.001			# temporal resolution of simulation
params['decay'] 	= 0.999			# decay at each time step
params['threshold'] = 1.0 			# membrane threshold
params['t_ref'] 	= 0.0			# Refractory period
params['alpha_ff'] 	= [0.5,0.5,0.5]	# synaptic scaling factors for feedforward weights
params['alpha_rec'] = [0.5]		# Synaptic scaling factor for recurrent weights
params['beta'] 		= [6., 6.5, 7.5]	# Synaptic threhsold scaling factors

# Inc, dec, beta, alpha_ff, and alpha_rec and sometimes max_rate

print("train acuracy")
print(model.evaluate(seqX_test, seqY_test))
SNN = sleepSNN([784,100,100,10], params)
numiterations = 10000 # simulation time
sleep_input = np.tile(np.mean(X_train,0), (numiterations,1))
# mean array of pixels across all dimensions repeated 10000 times
print(sleep_input.shape)
sleep_model = SNN.sleep(model, sleep_input, numiterations)

print("sleep acuracy")
print(sleep_model.evaluate(seqX_test, seqY_test))
