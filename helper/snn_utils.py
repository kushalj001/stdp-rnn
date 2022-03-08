import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.io import loadmat
from keras.optimizers import SGD, rmsprop

def create_tasks(y, num_tasks=5):
	'''
	returns a task label for incremental learning framework,
	y - one-hot encoded output labels
	'''

	num_examples = len(y)

	labels = np.argmax(y, 1)
	num_labels = len(np.unique(labels))
	labels_per_task = num_labels/num_tasks # 2

	assert isinstance(labels_per_task, int)

	class_labels = np.zeros((num_examples,))

	for i in range(num_examples):
		if labels[i] < labels_per_task: 
			class_labels[i] = 0 # 0 & 1
		elif labels[i] < 2*labels_per_task: 
			class_labels[i] = 1 # 2 & 3
		elif labels[i] < 3*labels_per_task:
			class_labels[i] = 2 # 4 & 5
		elif labels[i] < 4*labels_per_task:
			class_labels[i] = 3 # 6 & 7
		else:
			class_labels[i] = 4 # 8 & 9

	return class_labels

def build_FCN_model(lr=0.1, bias=True, architecture=[1200, 1200, 10], dropout=[0.2, 0.2, 0], num_features=(784,)):
	model = Sequential()
	for i in range(len(architecture)):
		if i == 0:
			model.add(Dense(architecture[i], activation='relu', use_bias=bias, input_shape=num_features))
		elif i < len(architecture) - 1:
			model.add(Dense(architecture[i], activation='relu', use_bias=bias))
		else:
			model.add(Dense(architecture[i], activation='softmax',use_bias=bias))

		if dropout[i] != 0:
			model.add(Dropout(dropout[i]))
	# [784, 1200] => [784, 1200] => [784,10]
	sgd = SGD(lr=lr, momentum=0.5, decay=0.0, nesterov=False)

	# Compile the model with crossentropy
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])

	return model

def load_MNIST_data():

	# Load pre-shuffled MNIST data into train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# X_train = [60k, 28, 28]
	# Reshape for Keras formatting (numchannels X length X Width)
	X_train = X_train.reshape(X_train.shape[0],784)
	# [N, 784]
	X_test = X_test.reshape(X_test.shape[0],784)
	print X_train.shape

	# Normalize between 0 and 1 for all inputs
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	# Convert 1-dimensional class arrays to 10-dimensional class matrices (one-hot encoding)
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)

	return X_train, Y_train, X_test, Y_test

def load_CUB200_data(filepath='/Users/timtadros/Documents/sleep-alg-keras/CUB200/'):
	training_data = loadmat(filepath+'cub200_resnet50_train.mat')
	test_data = loadmat(filepath+'cub200_resnet50_test.mat')
	train_x = training_data['X']
	test_x = test_data['X']
	train_y = training_data['y']
	test_y = test_data['y']
	
	train_y = train_y.reshape(train_y.shape[1], train_y.shape[0])
	test_y = test_y.reshape(test_y.shape[1], test_y.shape[0])

	train_y = np_utils.to_categorical(train_y, 200)
	test_y = np_utils.to_categorical(test_y, 200)

	# normalize
	train_x = train_x.astype('float32')
	test_x = test_x.astype('float32')
	train_x /= np.max(train_x)
	test_x /= np.max(train_x)

	return train_x, train_y, test_x, test_y

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_CIFAR10_data(filepath='/Users/timtadros/Documents/sleep-alg-keras/CIFAR10/'):
	# load training set
	train_x = np.zeros((50000, 3072))
	train_y = np.zeros((50000,))

	for i in range(5):
		file = unpickle(filepath+'data_batch_'+str(i+1))
		x = file['data']
		train_x[i*10000:(i+1)*10000,:] = file['data']
		train_y[i*10000:(i+1)*10000] = file['labels']

	testfile = unpickle(filepath+'test_batch')
	test_x = testfile['data']
	test_y = testfile['labels']	

	# normalize
	train_x = train_x.astype('float32')
	test_x = test_x.astype('float32')
	train_x /= 255
	test_x /= 255

	# Convert 1-dimensional class arrays to 10-dimensional class matrices (one-hot encoding)
	train_y = np_utils.to_categorical(train_y, 10)
	test_y = np_utils.to_categorical(test_y, 10)

	return train_x, train_y, test_x, test_y

def compute_task_accuracy(X, y, model, tasks):
	num_tasks = len(np.unique(tasks))
	accuracy = np.zeros((num_tasks,))
	for i in range(num_tasks):
		task_x = X[tasks==i]
		task_y = y[tasks==i]

		score = model.evaluate(task_x, task_y, verbose=0)
		accuracy[i] = score[1]
	return accuracy

def plot_accuracy_table(accuracy, num_tasks):
	fig = plt.figure()
	plt.imshow(accuracy)
	plt.colorbar()
	plt.show()

def build_CIFAR_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=(32,32,3)))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

def create_sleep_input(X, numexamples, masksize):
	# X = [N, 784] N ~ variable depending on the task number
	# Not sure what this is exactly. Passed as numiterations; 100 for the first
	# phase, 1100 for the second sleep phase and so on. Means that the sleep phase is run only on 
	# a fraction of the examples?

	# N != numexamples
	sleep_input = np.mean(X,0).reshape(28,28)
	# [N, 784] => [784] => [28,28], mean across every dimension
	sleep_X = np.zeros((numexamples, 28, 28))
	# [100, 28, 28] for first task
	for i in range(numexamples):
		x_pos = np.random.randint(28-masksize)
		y_pos = np.random.randint(28-masksize)
		sleep_X[i,x_pos:x_pos+masksize, y_pos:y_pos+masksize] = sleep_input[x_pos:x_pos+masksize,y_pos:y_pos+masksize]
		# randomly making some pixels in every image as the average pixel values.
	sleep_X = np.reshape(sleep_X, (numexamples, 784))
	return sleep_X


