import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from sleep import Sleep
from snn_utils import *
import sys

from snn_utils import create_sleep_input
from snn_utils import build_FCN_model
from snn_utils import create_tasks
from snn_utils import load_MNIST_data
from snn_utils import compute_task_accuracy
sys.path.append('/Users/timtadros/Documents/sleep-alg-keras/SleepCode')
from sleep import *

def plot_weights(model):
	inds = []
	for i in range(len(model.layers)):
		config = model.layers[i].get_config()
		if 'dense' in config['name']:
			inds.append(i)

	fig = plt.figure(figsize=(8,11))
	for i,ind in enumerate(inds):
		weights = model.layers[ind].get_weights()[0]
		plt.subplot(len(inds),1,i+1)
		plt.imshow(weights)
		plt.colorbar()
		plt.tight_layout()
	plt.show()

# Load the data
X_train, Y_train, X_test, Y_test = load_MNIST_data()

# Build the model
model = build_FCN_model(0.1, False, [1200, 1200, 10], [0.2, 0.2, 0], (784,))

# set up CF tasks (5 task setting)
task_labels_tr = create_tasks(Y_train, 5)
task_labels_te = create_tasks(Y_test, 5)

# Train each task sequentially for 2 epochs
num_examples=5000
num_iterations=100
masksize=10
num_epochs = 1
batch=100

accuracy = np.zeros((5,10))
ind = 0
for i in range(5):
	task_X = X_train[task_labels_tr==i,:][0:num_examples]
	# all the examples where the task label matches the idx
	# [10k, 784]
	task_Y = Y_train[task_labels_tr==i,:][0:num_examples]

	# Train the model
	model.fit(task_X, task_Y, 
          batch_size=batch, nb_epoch=num_epochs, verbose=1)
	
	# Evaluate performance on test set
	accuracy[:,ind] = compute_task_accuracy(X_test, Y_test, model, task_labels_te)
	plot_weights(model)
	sleep = Sleep(16, 0.001, 0.0001, 2.25, 1.0, 0, 0.999, 0.001)
	numiterations = num_iterations + (i*1000)
	# 100 for first task, 1100 for 2nd task, 2100,...
	sleep_input = create_sleep_input(X_train[task_labels_tr <= i,:], numiterations, masksize)
	# X_train[task_labels_tr <= i,:] - include training examples from the current task as well as
	# from the previous tasks. The first sleep phase is run only on the examples from 1st task,
	# i.e., model learns the difference between 0 and 1. For the second phase, the exmaples include 
	# task 1 as well as task 2 examples and so on. The final sleep phase runs on all the examples.

	# sleep_input = [numiterations, 784]
	model = sleep.run_sleep(model, sleep_input, numiterations, [0.95, 0.65, 0.35], [1.,1.,1.], use_bias=False)
	plot_weights(model)
	# Evaluate performance on test set
	accuracy[:,ind+1] = compute_task_accuracy(X_test, Y_test, model, task_labels_te)
	ind += 2

print accuracy
# plot table of accuracies
plot_accuracy_table(accuracy, 5)
