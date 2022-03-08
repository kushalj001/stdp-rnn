
from collections import defaultdict
from torch import nn
import numpy as np
import torch


def create_sleep_input(data, vectors, word2idx, num_iterations):
    all_tokens = []
    for ex in data:
        all_tokens.append(ex.text)
    example_means = []
    for ex_tokens in all_tokens:
        tmp = []
        for tok in ex_tokens:
            tmp.append(vectors[word2idx[tok]])
        tmp = torch.stack(tmp, dim=0)
        # print(tmp.shape)
        # [num_tokens, 100]
        example_means.append(torch.mean(tmp, 0))
    sleep_input = torch.stack(example_means, 0)
    sleep_input = torch.mean(sleep_input, 0)
    sleep_input = torch.tile(sleep_input, (num_iterations,1))
    # [10000, 100]
    return sleep_input

class SNNLayer:
    def __init__(self, num_iterations, num_neurons):
        self.mem = np.zeros(num_neurons)
        self.refractory_end = np.zeros(num_neurons)
        self.sum_spikes = np.zeros(num_neurons)
        self.spikes = np.zeros(num_neurons)
        self.all_spikes = np.zeros((num_iterations, num_neurons))


class SleepRNNLayer:
    def __init__(self, layer_sizes, params):
        self.inc = params["inc"]
        self.dec = params["dec"]
        self.max_rate = params["max_rate"]
        self.dt = params["dt"]
        self.decay = params["decay"]
        self.threshold = params["threshold"]
        self.t_refractory = params["t_refractory"]
        self.alpha_linear = params["alpha_linear"]
        self.alpha_rec = params["alpha_rec"]
        self.beta = params["beta"]
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
    
    def sigmoid(self, x):
        y = 2 * ( 1.0 - ( 1.0 / ( 1.0 + np.exp( -x ) ) ) )
        return y
        
    def initialize_layers(self, num_iterations):
        layers = []
        for i in range(self.num_layers):
            layers.append(SNNLayer(num_iterations, self.layer_sizes[i]))
        return layers
    
    def get_weights(self, model):
        recurrent_weights = []
        linear_weights = []
        is_recurrent = []
        layer_idx_mapping = defaultdict(list)
        idx = 0
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                layer_object = getattr(model, name)
                linear_weights.append(layer_object.weight.T.cpu().detach().numpy())
                layer_idx_mapping[name].append(idx)
                is_recurrent.append(False)
                idx += 1
            
            elif isinstance(layer, nn.RNN):
                layer_object = getattr(model, name)
                linear_weights.append(layer_object.weight_ih_l0.T.cpu().detach().numpy())
                layer_idx_mapping[name].append(idx)
                recurrent_weights.append(layer_object.weight_hh_l0.cpu().detach().numpy())
                layer_idx_mapping[name].append(idx)
                is_recurrent.append(True)
                idx += 1
        return linear_weights, recurrent_weights, is_recurrent, layer_idx_mapping
    
    def set_weights(self, model, linear_weights, recurrent_weights, layer_idx_mapping):
        for name, module in model.named_modules():
            if name not in layer_idx_mapping:
                continue
            idx = layer_idx_mapping[name]
            if len(idx) > 1:
                layer_object = getattr(model, name)
                layer_object.weight_ih_l0 = nn.Parameter(torch.from_numpy(linear_weights[idx[0]].T))
                layer_object.weight_hh_l0 = nn.Parameter(torch.from_numpy(recurrent_weights[idx[1]]))
            else:
                layer_object = getattr(model, name)
                layer_object.weight = nn.Parameter(torch.from_numpy(linear_weights[idx[0]]))
        return model

    def sleep(self, model, sleep_input, num_iterations):

        _, num_features = sleep_input.shape
        self.layers = self.initialize_layers(num_iterations)
        linear_weights, recurrent_weights, is_recurrent, layer_idx_mapping = self.get_weights(model)

        for t in range(num_iterations):
            rescale_factor = 1.0 / (self.dt * self.max_rate)
            spike_snapshot = np.random.uniform(0.0, 1.0, (num_features,)) * rescale_factor
            inp = spike_snapshot <= sleep_input[t,:]
            
            self.layers[0].spikes = inp
            self.layers[0].sum_spikes += inp
            self.layers[0].all_spikes[t,:] = inp
            
            recl = 0
            # [(100, 256), (256, 128), (128,1)]
            # [(256, 256)]
            # [100, 256, 128, 1]
            for l in range(1, self.num_layers):
                impulse = self.alpha_linear[l-1] * linear_weights[l-1].T @ self.layers[l-1].spikes
                # l=1: [256, 100].[100] => [256]; weights of the first layer with the input
                # l=2: [128, 256].[256] => [128]
                # l=3: [1, 128].[128] => [1]
                
                if is_recurrent[l-1]:
                    impulse = np.add(impulse, self.alpha_rec[recl] * recurrent_weights[recl].T @ self.layers[l].spikes)
                    # [256, 256] @ [256] = 256
                    # alpha_rec[0] * recurrent_weights[0].T @ layers[1].spikes (always).
                    # evaluates to a non-zero value after t >= 1. At t=0, layers[1].spikes = 0
                
                self.layers[l].mem = np.add(self.decay * self.layers[l].mem, impulse)
                self.layers[l].spikes = self.layers[l].mem >= self.threshold * self.beta[l-1]
                
                # STDP
                pre = np.where(self.layers[l-1].spikes == 1)[0]
                post = np.where(self.layers[l].spikes == 1)[0]
                notpre = np.where(self.layers[l-1].spikes == 0)[0]
                
                w = linear_weights[l-1]
                w[pre[:,None], post] += self.inc * self.sigmoid(w[pre[:,None],post])
                w[notpre[:,None], post] -= self.dec * self.sigmoid(w[notpre[:,None],post])
                
                
                # Recurrent STDP
                if is_recurrent[l-1] and t > 1:
                    post = np.where(self.layers[l].spikes == 1)[0]
                    pre = np.where(self.layers[l].all_spikes[t-1,:] == 1)[0]
                    notpre = np.where(self.layers[l].all_spikes[t-1,:] == 0)[0]
                    # Notion of presynaptic and post-synaptic neurons for the recurrent weight comes 
                    # from time. The same weight matrix gets updated at every time step.
                    w = recurrent_weights[recl]
                    w[pre[:,None], post] += self.inc  * self.sigmoid(w[pre[:,None], post])
                    w[notpre[:,None], post] -= self.dec * self.sigmoid(w[notpre[:,None], post])
                    recl += 1
                
                
                self.layers[l].refractory_end[np.where(post==1)[0]] = t + self.t_refractory
                self.layers[l].mem[post] = 0.
                
                self.layers[l].sum_spikes = self.layers[l].sum_spikes + self.layers[l].spikes
                self.layers[l].all_spikes[t,:] = self.layers[l].spikes
        
        model = self.set_weights(model, linear_weights, recurrent_weights, layer_idx_mapping)
        return model