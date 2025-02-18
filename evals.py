import tensorflow as tf
import matplotlib.pyplot as plt
from problems.adding_problem_pytorch import AddingProblemDataset
from models.tf_rnn import TFRNN
from models.urnn_cell import URNNCell
import numpy as np

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
        optimizer,
        loss_function):
'''

loss_path = 'results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def baseline_cm(timesteps):
    return 10 * np.log(8) / timesteps

def baseline_ap():
    return 0.167

def serialize_loss(loss, name):
    with open(loss_path + name, 'w') as file:
        for l in loss:
            file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        # init adding problem
        self.ap_batch_size = 50
        self.ap_epochs = 15

        self.ap_timesteps = [100, 200, 400, 750]
        self.ap_samples = [30000, 50000, 40000, 100000]
        self.ap_data = [AddingProblemDataset(sample, timesteps) for 
                        timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
        self.dummy_ap_data = AddingProblemDataset(100, 50)  # samples, timesteps

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=', sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')

    def train_urnn_for_timestep_idx(self, idx):
        print('Initializing and training URNNs for one timestep...')

        # AP

        # No need for tf.reset_default_graph() in TensorFlow 2.x

        self.ap_urnn = TFRNN(
            name="ap_urnn",
            num_in=2,
            num_hidden=512,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=URNNCell,
            activation_hidden=None,  # modReLU
            activation_out=tf.identity,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=glob_learning_rate, rho=glob_decay),
            loss_function=tf.math.square)
        self.train_network(self.ap_urnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training URNNs for one timestep done.')

    def train_rnn_lstm_for_timestep_idx(self, idx):
        print('Initializing and training RNN & LSTM for one timestep...')

        # AP

        # No need for tf.reset_default_graph() in TensorFlow 2.x

        from tensorflow.keras.layers import SimpleRNNCell, LSTMCell

        self.ap_simple_rnn = TFRNN(
            name="ap_simple_rnn",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=SimpleRNNCell,
            activation_hidden=tf.nn.tanh,
            activation_out=tf.identity,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=glob_learning_rate, rho=glob_decay),
            loss_function=tf.math.square)
        self.train_network(self.ap_simple_rnn, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        # LSTM cell
        self.ap_lstm = TFRNN(
            name="ap_lstm",
            num_in=2,
            num_hidden=128,
            num_out=1,
            num_target=1,
            single_output=True,
            rnn_cell=LSTMCell,
            activation_hidden=tf.nn.tanh,
            activation_out=tf.identity,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=glob_learning_rate, rho=glob_decay),
            loss_function=tf.math.square)
        self.train_network(self.ap_lstm, self.ap_data[idx], 
                           self.ap_batch_size, self.ap_epochs)

        print('Init and training networks for one timestep done.')

    def train_networks(self):
        print('Starting training...')

        timesteps_idx = 4
        for i in range(timesteps_idx):
            self.train_urnn_for_timestep_idx(i)
            fig = plt.figure()

            # plot loss
            ax = fig.add_subplot(111)
            ax.plot(self.ap_urnn.get_loss_list())
            ax.set_title('Loss for uRNN with ' + str(self.ap_timesteps[i]) + ' timesteps')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            plt.show()
        for i in range(timesteps_idx):
            self.train_rnn_lstm_for_timestep_idx(i)
            fig = plt.figure()

            # plot loss
            ax = fig.add_subplot(111)
            ax.plot(self.ap_lstm.get_loss_list())
            ax.set_title('Loss for LSTM with ' + str(self.ap_timesteps[i]) + ' timesteps')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            plt.show()

        print('Done and done.')

if __name__ == '__main__':
    main = Main()
    main.init_data()
    main.train_networks()