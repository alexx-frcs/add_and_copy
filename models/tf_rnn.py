import numpy as np
import tensorflow as tf
from .urnn_cell import URNNCell

def serialize_to_file(loss):
    with open('losses.txt', 'w') as file:
        for l in loss:
            file.write("{0}\n".format(l))

class TFRNN(tf.keras.Model):
    def __init__(
        self,
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

        super(TFRNN, self).__init__(name=name)

        # self
        self.name = name
        self.loss_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        self.log_dir = './logs/'

        # Optimizer and loss function
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.activation_out = activation_out
        self.single_output = single_output
        self.num_target = num_target

        # Initialize RNN cell
        if rnn_cell == URNNCell:
            self.cell = rnn_cell(num_units=num_hidden, num_in=num_in)
        elif rnn_cell == tf.keras.layers.LSTMCell:
            self.cell = rnn_cell(num_hidden)
        elif rnn_cell == tf.keras.layers.SimpleRNNCell:
            self.cell = rnn_cell(num_hidden, activation=activation_hidden)
        else:
            self.cell = rnn_cell(num_units=num_hidden, activation=activation_hidden)

        # Extract output size
        self.output_size = self.cell.output_size
        if isinstance(self.output_size, dict):
            self.output_size = self.output_size['num_units']

        # Define RNN layer
        self.rnn = tf.keras.layers.RNN(
            self.cell,
            return_sequences=not single_output,
            return_state=True
        )

        # Set up h->o parameters (weights and biases)
        self.w_ho = self.add_weight(
            name="w_ho_"+self.name,
            shape=(self.output_size, num_out),
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b_o = self.add_weight(
            name="b_o_"+self.name,
            shape=(num_out,),
            initializer='zeros'
        )

        # Number of trainable parameters
        t_params = np.sum([np.prod(v.shape) for v in self.trainable_variables])
        print('Network __init__ over. Number of trainable params=', t_params)

    def call(self, inputs, training=False, initial_state=None):
        # inputs: [batch_size, time_steps, num_in]
        outputs_h, *final_state = self.rnn(
            inputs,
            initial_state=initial_state,
            training=training
        )

        # Produce final outputs from hidden layer outputs
        if self.single_output:
            outputs_h_last = outputs_h  # [batch_size, self.output_size]
            preact = tf.matmul(outputs_h_last, self.w_ho) + self.b_o
            outputs_o = self.activation_out(preact)  # [batch_size, num_out]
        else:
            # outputs_h: [batch_size, time_step, self.output_size]
            preact = tf.tensordot(outputs_h, self.w_ho, axes=[[2], [0]]) + self.b_o
            outputs_o = self.activation_out(preact)  # [batch_size, time_step, num_out]

        return outputs_o, final_state

    def compute_loss(self, outputs_o, targets):
        # Assurez-vous que outputs_o et targets sont du même type
        outputs_o = tf.cast(outputs_o, tf.float32)
        targets = tf.cast(targets, tf.float32)
        if self.loss_function == tf.square:
            loss = tf.reduce_mean(tf.square(outputs_o - targets))
        elif self.loss_function == tf.nn.sparse_softmax_cross_entropy_with_logits:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            targets = tf.cast(tf.squeeze(targets), tf.int32)
            loss = loss_fn(targets, outputs_o)
        else:
            raise Exception('Unsupported loss function')
        return loss

    @tf.function
    def train_step(self, X, Y):
        batch_size = tf.shape(X)[0]
        initial_state = self.get_initial_state(batch_size)
        with tf.GradientTape() as tape:
            outputs_o, _ = self(X, training=True, initial_state=initial_state)
            loss = self.compute_loss(outputs_o, Y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def evaluate(self, X, Y):
        batch_size = X.shape[0]
        initial_state = self.get_initial_state(batch_size)
        outputs_o, _ = self(X, training=False, initial_state=initial_state)
        loss = self.compute_loss(outputs_o, Y)
        return loss.numpy()

    def get_initial_state(self, batch_size):
        # Return initial_state as per cell.state_size
        if isinstance(self.cell.state_size, int):
            # state_size is an integer
            state = tf.random.uniform(
                [batch_size, self.cell.state_size],
                -self.init_state_C,
                self.init_state_C
            )
        elif isinstance(self.cell.state_size, (tuple, list)):
            # LSTMStateTuple or list
            state = [
                tf.random.uniform(
                    [batch_size, s],
                    -self.init_state_C,
                    self.init_state_C
                ) for s in self.cell.state_size
            ]
        else:
            raise ValueError('Unsupported state_size type')
        return state

    def train(self, dataset, batch_size, epochs):
        # Fetch validation data
        X_val, Y_val = dataset.get_validation_data()

        # Initialize loss list
        self.loss_list = []
        print("Starting training for", self.name)
        num_batches = dataset.get_batch_count(batch_size)
        print("NumEpochs:", '{0:3d}'.format(epochs), 
              "|BatchSize:", '{0:3d}'.format(batch_size), 
              "|NumBatches:", '{0:5d}'.format(num_batches),'\n')

        for epoch_idx in range(epochs):
            print("Epoch Starting:", epoch_idx, '\n')

            for batch_idx in range(num_batches):
                X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)
                batch_loss = self.train_step(X_batch, Y_batch)
                self.loss_list.append(batch_loss.numpy())

                if batch_idx % 10 == 0:
                    total_examples = (
                        batch_size * num_batches * epoch_idx
                        + batch_size * batch_idx
                        + batch_size
                    )
                    # Serialize loss
                    serialize_to_file(self.loss_list)
                    print("Epoch:", '{0:3d}'.format(epoch_idx), 
                          "|Batch:", '{0:3d}'.format(batch_idx), 
                          "|TotalExamples:", '{0:5d}'.format(total_examples), 
                          "|BatchLoss:", '{0:8.4f}'.format(batch_loss))

            # Validate after each epoch
            validation_loss = self.evaluate(X_val, Y_val)
            mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
            print("Epoch Over:", '{0:3d}'.format(epoch_idx), 
                  "|MeanEpochLoss:", '{0:8.4f}'.format(mean_epoch_loss),
                  "|ValidationSetLoss:", '{0:8.4f}'.format(validation_loss),'\n')

    def test(self, dataset):
        # Fetch test set
        X_test, Y_test = dataset.get_test_data()
        test_loss = self.evaluate(X_test, Y_test)
        print("Test set loss:", test_loss)

    # Loss list getter
    def get_loss_list(self):
        return self.loss_list