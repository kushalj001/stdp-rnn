import numpy as np

class SNNLayer:
	def __init__(self, num_neurons, numiterations):
		self.mem = np.zeros(num_neurons,)
		self.refrac_end = np.zeros(num_neurons,)
		self.sum_spikes = np.zeros(num_neurons,)
		self.spikes = np.zeros(num_neurons,)
		self.total_spikes = np.zeros((numiterations, num_neurons))


class sleepSNN:
	def __init__(self, size, params):
		# size = [784,100,100,10]
		self.inc = params['inc']		# Magnitude of weight increase upon STDP event
		self.dec = params['dec']	# Magnitude of weight decrease upon STDP event
		self.max_rate = params['max_rate']	# Maximum firing rate of neurons in the input layer
		self.dt = params['dt']	# temporal resolution of simulation
		self.decay = params['decay']	# decay at each time step
		self.threshold = params['threshold'] # membrane threshold
		self.t_ref = params['t_ref']	# Refractory period
		self.alpha_ff = params['alpha_ff']	# synaptic scaling factors for feedforward weights
		self.alpha_rec = params['alpha_rec']	# Synaptic scaling factor for recurrent weights
		self.beta = params['beta']		# firing threshold scaling factor
		self.num_layers = len(size)
		self.num_neurons = size

	def initialize_layers(self, numiterations):
		'''
		Initialize layers in the spiking neural network based on size of the RNN.
		'''
		layers = []
		for i in range(self.num_layers):
			layers.append(SNNLayer(self.num_neurons[i], numiterations))
		return layers

	def sigmoid(self, x):
		y = 2 * ( 1.0 - ( 1.0 / ( 1.0 + np.exp( -x ) ) ) )
		return y

	def set_weights(self, model, ff_weights, rec_weights, rec_layers):
		rec_counter = 0
		ff_counter = 0
		for i in range(len(rec_layers)):
			if rec_layers[i]:
				model.layers[i].set_weights([ff_weights[ff_counter], rec_weights[rec_counter]])
				ff_counter += 1
				rec_counter += 1
			else:
				model.layers[i].set_weights([ff_weights[ff_counter]])
				ff_counter += 1
		return model

	def get_weights(self, model):
		rec_weights = []
		ff_weights = []
		rec_layers = []
		for i,layer in enumerate(model.layers):
			w = layer.get_weights()
			if len(w) == 2:
				# use_bias = False; hence the length of weight matrices is 2
				# It contains W_xh which is input to hidden weights and
				# W_hh which is weight matrix for hidden to hidden transformation.
				# h_t = f(x_t*W_xh + h_t-1*W_hh)
				# The dimention of W_hh is the number of units provided initially.
				rec_layers.append(True)
				ff_weights.append(w[0])
				# W_xh weights
				rec_weights.append(w[1])
				# W_hh weights
			elif len(w) == 1:
				# Normal dense layers
				ff_weights.append(w[0])
				rec_layers.append(False)
			else:
				continue

		return ff_weights, rec_weights, rec_layers

	def sleep(self, model, sleep_input, numiterations):
		# No weight normalization is done for the SNN in this model. 
		# There are 2 ways of normalizing the weights of SNN after
		# transferring them from ANN to SNN to ensure loss-less conversion: model-based and data-based.
		# Once the normalization strategy is determined, we run a grid search for the firing rate and 
		# spiking threshold for the best model.

		# sleep_input = [10000, 784]
		# numiterations = 10000
		numiterations, num_features = sleep_input.shape
		self.layers =  self.initialize_layers(numiterations) 
		# [784, 100, 100, 10]
		ff_weights, rec_weights, rec_layers = self.get_weights(model)
		# ff_weights = [(784, 100), (100, 100), (100, 10)]
		# rec_weights = [(100,100)]
		# rec_layers = [True, False, False]
		num_layers = len(ff_weights)
		for t in range(numiterations):
			# Create input poisson train
			rescale_fact = 1.0/(self.dt*self.max_rate)
			# 1 / (0.001 * 32) = 31.25
			# The max_rate is a hyperparam that is decided by doing a grid search over some values
			# along with the spiking threshold of neurons.
			spike_snapshot = np.random.uniform(0.0, 1.0, (num_features,)) * rescale_fact
			# [784]
			inp_image = spike_snapshot <= sleep_input[t,:]
			self.layers[0].spikes = inp_image
			self.layers[0].sum_spikes = self.layers[0].sum_spikes + inp_image
			self.layers[0].total_spikes[t,:] = self.layers[0].spikes
			# total spikes records or keeps track of spikes across the time dimension.
			# Put simply, it stores spikes for all the iterations. The other
			# attributes of the layer only record the state of the neurons/layer at the current instant.
			recl = 0 

			for l in range(1, self.num_layers):
				# get impulse from incoming spikes		
				impulse = self.alpha_ff[l-1] * ff_weights[l-1].T @ self.layers[l-1].spikes
				# l = 1: [100, 784] @ [784] => [100]

				# check if recurrent layer and add impulse from recurrent layer
				if rec_layers[l-1] == True:
					impulse = np.add(impulse, self.alpha_rec[recl] * rec_weights[recl].T @ self.layers[l].spikes)
					# l=1: [100,100] @ [100] => [100]; wouldn't the spikes be zero for the first layer 
					# initially?
				# Add input to membrane potential and check for spikes
				self.layers[l].mem = np.add(self.decay * self.layers[l].mem, impulse)
				self.layers[l].spikes = self.layers[l].mem >= self.threshold * self.beta[l-1]

				# STDP
				pre = np.where(self.layers[l-1].spikes == 1)[0]
				post = np.where(self.layers[l].spikes == 1)[0]
				notpre = np.where(self.layers[l-1].spikes == 0)[0]
				w = ff_weights[l-1]
				if l-1 > 0:
					w[pre,post[:,np.newaxis]] += self.inc * self.sigmoid(w[pre,post[:,np.newaxis]])
					w[notpre,post[:,np.newaxis]] -= self.dec * self.sigmoid(w[notpre,post[:,np.newaxis]])
				else:
					w[pre,post[:,np.newaxis]] += self.inc * self.sigmoid(w[pre,post[:,np.newaxis]])
					w[notpre,post[:,np.newaxis]] -= self.dec * self.sigmoid(w[notpre,post[:,np.newaxis]])
				

				# Recurrent STDP
				if rec_layers[l-1] == True and t > 1:
					post = np.where(self.layers[l].spikes == 1)[0]
					pre  = np.where(self.layers[l].total_spikes[t-1,:] == 1)[0]
					notpre = np.where(self.layers[l].total_spikes[t-1,:] == 0)[0]
					w = rec_weights[recl]
					w[post,pre[:,np.newaxis]] += self.inc * self.sigmoid(w[post,pre[:,np.newaxis]])
					w[post,notpre[:,np.newaxis]] -= self.dec * self.sigmoid(w[post,notpre[:,np.newaxis]])
					recl += 1
				
				# refractory period
				self.layers[l].refrac_end[np.where(post == 1)[0]] = t + self.t_ref

				# reset membrane potential
				self.layers[l].mem[post] = 0.0
				# the neurons that spiked have to reset to 0.
				# after action potential the neuron's potential goes below the 
				# resting potential for some time and then reaches the resting potential again.

				# Update layer statistics
				self.layers[l].sum_spikes = self.layers[l].sum_spikes + self.layers[l].spikes
				self.layers[l].total_spikes[t,:] = self.layers[l].spikes

		model = self.set_weights(model, ff_weights, rec_weights, rec_layers)
		return model