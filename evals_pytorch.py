# evals_pytorch.py (Version PyTorch corrigée)

import torch
import matplotlib.pyplot as plt
from problems.adding_problem_pytorch import AddingProblemDataset
from models_pytorch.tf_rnn_pytorch import TFRNN
from models_pytorch.urnn_cell_pytorch import URNNCell
from models_pytorch.lru_cell_pytorch import LRUCell
from models_pytorch.my_RNN_pytorch import SimpleRNNCell
from models_pytorch.new_LRU_pytorch import my_LRUCell
import numpy as np
import torch.nn as nn
import os

'''
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer_class,
        loss_function,
        learning_rate,
        decay
'''

loss_path = 'results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def baseline_cm(timesteps):
    return 10 * np.log(8) / timesteps

def baseline_ap():
    return 0.167

def serialize_loss(loss, name):
    os.makedirs(os.path.dirname(loss_path + name), exist_ok=True)
    with open(loss_path + name, 'w') as file:
        for l in loss:
            file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        # Init Adding Problem
        self.ap_batch_size = 50
        self.ap_epochs = 15

        # self.ap_timesteps = [100, 200, 400, 750]
        self.ap_timesteps = [100, 750]
        # self.ap_samples = [30000, 50000, 40000, 100000]
        self.ap_samples = [30000, 100000]
        self.ap_data = [AddingProblemDataset(sample, timesteps) for 
                        timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
        self.dummy_ap_data = AddingProblemDataset(100, 50)  # samples, timesteps

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.sample_len)
        print('Training network ', net.name, '... timesteps=', sample_len)
        net.train_network(dataset, batch_size, epochs)  # Méthode adaptée pour PyTorch
        # loss_list contient un nombre par batch (step)
        serialize_loss(net.get_loss_list(), net.name + f"_{sample_len}.txt")
        print('Training network ', net.name, ' done.')

    def train_urnn_for_timestep_idx(self, idx):
        print('Initializing and training URNNs for one timestep...')

        # AP

        # Initialisation du modèle URNN
        self.ap_urnn = TFRNN(
            name="ap_urnn",
            rnn_cell=URNNCell,
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            activation_hidden=None,  # modReLU (géré dans URNNCell)
            activation_out=lambda x: x,  # identity
            optimizer_class=torch.optim.RMSprop,
            loss_function='mse',
            learning_rate=glob_learning_rate,
            decay=glob_decay,
            timesteps=self.ap_timesteps[idx], 
            n_filters=20
        )
        self.train_network(self.ap_urnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training URNNs for one timestep done.')
    
    def train_lru_for_timestep_idx(self, idx, n_filters):
        print('Initializing and training LRU for one timestep...')

        # AP

        # Initialisation du modèle LRU
        # self.ap_lru = TFRNN(
        #     name="ap_urnn",
        #     rnn_cell=LRUCell,
        #     num_in=2,
        #     num_hidden=512,
        #     num_out=1,
        #     num_target=1,
        #     single_output=True,
        #     activation_hidden=None,  
        #     activation_out=torch.tanh,  # tanh
        #     optimizer_class=torch.optim.RMSprop,
        #     loss_function='mse',
        #     learning_rate=glob_learning_rate,
        #     decay=glob_decay, 
        #     timesteps=self.ap_timesteps[idx], 
        #     n_filters=n_filters
        # )
        # self.train_network(self.ap_lru, self.ap_data[idx], 
        #                    self.ap_batch_size, self.ap_epochs)
        
        # Initialisation du modèle LRU
        self.ap_lru = TFRNN(
            name="ap_lru",
            rnn_cell=my_LRUCell,
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            activation_hidden=None,  
            activation_out=lambda x: x,  # identity
            optimizer_class=torch.optim.RMSprop,
            loss_function='mse',
            learning_rate=glob_learning_rate,
            decay=glob_decay, 
            timesteps=self.ap_timesteps[idx], 
            n_filters=n_filters
        )
        self.train_network(self.ap_lru, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training LRU for one timestep done.')

    def train_rnn_lstm_for_timestep_idx(self, idx):
        print('Initializing and training RNN & LSTM for one timestep...')

        # AP

        # Initialisation Simple RNN
        self.ap_my_rnn = TFRNN(
            name="ap_my_rnn",
            rnn_cell=SimpleRNNCell,
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            activation_hidden=torch.tanh,
            activation_out=lambda x: x,  # identity
            optimizer_class=torch.optim.RMSprop,
            loss_function='mse',
            learning_rate=glob_learning_rate,
            decay=glob_decay,
            timesteps=self.ap_timesteps[idx], 
            n_filters=20
        )
        self.train_network(self.ap_my_rnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        # Initialisation pre implem RNN
        self.ap_RNN = TFRNN(
            name="ap_RNN",
            rnn_cell=nn.RNNCell,
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            activation_hidden=torch.tanh,
            activation_out=lambda x: x,  # identity
            optimizer_class=torch.optim.RMSprop,
            loss_function='mse',
            learning_rate=glob_learning_rate,
            decay=glob_decay,
            timesteps=self.ap_timesteps[idx], 
            n_filters=20
        )
        self.train_network(self.ap_RNN, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)
        # Initialisation LSTM
        self.ap_lstm = TFRNN(
            name="ap_lstm",
            rnn_cell=nn.LSTMCell,
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            activation_hidden=torch.tanh,
            activation_out=lambda x: x,  # identity
            optimizer_class=torch.optim.RMSprop,
            loss_function='mse',
            learning_rate=glob_learning_rate,
            decay=glob_decay,
            timesteps=self.ap_timesteps[idx], 
            n_filters=20
        )
        self.train_network(self.ap_lstm, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training networks for one timestep done.')

    def train_networks(self):
        print('Starting training...')

        timesteps_idx = 1
        for i in range(timesteps_idx):
            self.train_lru_for_timestep_idx(i, n_filters=5)
            fig = plt.figure()

            # Plot loss
            ax = fig.add_subplot(111)
            ax.plot(self.ap_lru.get_loss_list())
            ax.set_title('Loss for LRU with ' + str(self.ap_timesteps[i]) + ' timesteps')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            plt.show()
        for i in range(timesteps_idx):
            self.train_urnn_for_timestep_idx(i)
            fig = plt.figure()

            # Plot loss
            ax = fig.add_subplot(111)
            ax.plot(self.ap_urnn.get_loss_list())
            ax.set_title('Loss for uRNN with ' + str(self.ap_timesteps[i]) + ' timesteps')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            plt.show()
        # for i in range(timesteps_idx):
        #     self.train_rnn_lstm_for_timestep_idx(i)
        #     fig = plt.figure()

        #     # Plot loss with values capped above 1
        #     ax = fig.add_subplot(111)
        #     loss_list = np.clip(self.ap_my_rnn.get_loss_list(), None, 1)  # Cap values above 1
        #     ax.plot(loss_list, label='My RNN')
        #     loss_list2 = np.clip(self.ap_RNN.get_loss_list(), None, 1)
        #     ax.plot(loss_list2, label='RNN')
        #     loss_list3 = np.clip(self.ap_lstm.get_loss_list(), None, 1)
        #     ax.plot(loss_list3, label='LSTM')
        #     ax.set_title('Loss with ' + str(self.ap_timesteps[i]) + ' timesteps')
        #     ax.legend()
        #     ax.set_xlabel('Batch')
        #     ax.set_ylabel('Loss')
        #     ax.set_yscale('log')
        #     plt.show()

        print('Done and done.')

if __name__ == '__main__':
    main = Main()
    main.init_data()
    main.train_networks()
