import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Sleep: # 16, 0.001, 0.0001, 2.25, 1.0, 0, 0.999, 0.001
	def __init__(self, maxrate, inc, dec, alpha, beta, decay, Winh, dt):
		self.inc 	= inc  # 0.001
		self.dec 	= dec # 0.0001
		self.alpha 	= alpha #2.25
		self.thresholds = beta # 1.0
		self.decay 	= decay # 0
		self.maxrate = maxrate # 16
		self.dt 	= dt # 0.001
		self.Winh 	= Winh # 0.999
		self.t_ref = 0

	def run_sleep(self, model, sleep_input, numiterations, beta_scale, alpha_scale, use_bias=True):
		# numiterations = variable depending on the task number,
		# 100 for task 1, 1100 for task 2, 2100 for task 3.
		# model => # [784, 1200] => [784, 1200] => [784,10]
		# sleep_input = [numiterations, 784]
		# beta_scale = [0.95, 0.65, 0.35], 
		# alpha_scale = [1.,1.,1.],
		num_layers = len(model.layers)
		input_size = sleep_input.shape[1]
		# 784
		layer_inds = self.get_rel_layers(model)
		# [0,1,2]
		W = []
		b = []
		
		# Get weight and bias matrices
		for i in layer_inds:
			W.append(model.layers[i].get_weights()[0])
			# weight matrix
			if use_bias:
				b.append(model.layers[i].get_weights()[1])
		# W => [(784, 1200), (784, 1200), (784,10)]
		# Create arrays for sleep analysis
		membrane_potential 	= [np.zeros(input_size)]
		refrac_end 			= [np.zeros(input_size)]
		sum_spikes 			= [np.zeros(input_size)]
		total_spikes 		= [np.zeros((numiterations,input_size))]
		spikes        		= [np.zeros(input_size)]
		for i in layer_inds:
			membrane_potential.append(np.zeros(model.layers[i].get_weights()[0].shape[1]))
			refrac_end.append(np.zeros(model.layers[i].get_weights()[0].shape[1]))
			sum_spikes.append(np.zeros(model.layers[i].get_weights()[0].shape[1]))
			total_spikes.append(np.zeros((numiterations, model.layers[i].get_weights()[0].shape[1])))
			spikes.append(np.zeros(model.layers[i].get_weights()[0].shape[1]))

		# all except total_spikes: [784, 1200, 1200, 10]
		# total_spikes = [(100, 784), (100, 1200), (100, 1200), (100,10)]

		# Run sleep algorithm
		for t in range(numiterations):
			# Create Poisson input
			rescale_fac = 1./(self.maxrate*self.dt) # 1/(16 * 0.001) = 1000/16 = 62.5
			# What does maxrate signify?
			spike_snapshot = np.random.random(input_size) * rescale_fac/2.0
			# [784] * 31.25 (why this?)
			inp_image = spike_snapshot <= sleep_input[t,:]
			# [784], comparing the snapshot with the current input index
			# this creates a binary vector with 1 where the condition satisfies and 0 o.w.
			# Can think of max_rate and this img creation as converting a continuous valued
			# input image to binary valued input. The max_rate determines the maximum relu activation
			# that was manifested during the ANN training.
			sum_spikes[0] += inp_image
			total_spikes[0][t,:] = inp_image
			spikes[0] = inp_image

			# Propagate activity
			for i in range(len(layer_inds)):
				# check input current
				impulse = self.alpha * alpha_scale[i-1] * np.dot(np.transpose(W[i]),np.transpose(spikes[i]))
				# 2.25 * 1* [1200, 784].[784] => 2.25 * [1200] => [1200]
				# Essentially multiplying the input with the first set of weights matrix to 
				# get the current. Analogous to ANNs.
				# alpha_scale[i-1] for i=0 is the last value. What does alpha signify?
				# Current from the input to the first layer, hence 1200 values

				impulse = impulse - np.sum(impulse)/len(impulse) * self.Winh
				# standardize impulse, (why winH?)

				# update membrane potential
				membrane_potential[i+1] = self.decay * membrane_potential[i+1] + impulse
				# Decay the current membrane potential and add the current to it
				# length of membrane potenial is 4 and layer_inds is 3. Hence we can access i+1.
				# Check for spiking
				spikes[i+1] = membrane_potential[i+1] >= self.thresholds * beta_scale[i-1]
				# A neuron spikes whenever the membrane potential exceeds the neuron threshold.
				# What is beta_scale? How are the thresholds decided for each neuron?

				# Run STDP rule
				pre = np.where(spikes[i] == 1)[0]
				# indices of neurons which fire in the current layer; pre-synaptic neurons
				post = np.where(spikes[i+1] == 1)[0]
				# indices of neurons fire when input is propagated from the current layers
				# post-synaptic spiking neurons
				nopre = np.where(spikes[i] == 0)[0]
				# Neurons that do not fire
				if len(pre) > 0 and len(post) > 0:
					W[i][pre[:,None],post] = W[i][pre[:,None],post] + self.inc #* self.sigmoid(W[i][pre[:,None],post])
					# long-term potentiation
				if len(post) > 0:
					W[i][nopre[:,None],post] = W[i][nopre[:,None],post] - self.dec #* self.sigmoid(W[i][nopre[:,None],post])
					# long-term depression
				# Reset
				membrane_potential[i+1][post] = 0.0
				# reset to 0

				# Refractory period
				refrac_end[i+1][post] = t + self.t_ref

				# Store results
				sum_spikes[i+1] += spikes[i+1]
				total_spikes[i+1][t,:] = spikes[i+1]

		for i in range(len(layer_inds)):
			model.layers[layer_inds[i]].set_weights([W[i]])
		
		return model

	def sigmoid(self, x):
		return 2 * ( 1.0 - ( 1.0 / np.exp( -x / 0.001 ) ) )

	def get_rel_layers(self, model):
		# len(model.layers) = 3
		# len(inds) = 3; 
		inds = []
		for i in range(len(model.layers)):
			config = model.layers[i].get_config()
			if 'dense' in config['name']:
				inds.append(i)
	
		return inds
		# inds = [0,2,4]
	def normalize_weights(self, model, p=99.0):
		inds = self.get_rel_layers(model)
		alpha = np.zeros(len(inds))
		for i in inds:
			output = model.layers[i].output
			#print output
	
			num_neurons = len(output)
			neuron_index = int(p*num_neurons/100.0)
	
			sorted_activations = np.sort(output)
			alpha[i] = sorted_activations[neuron_index]
		return alpha
