import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Diagonal unitary matrix
class DiagonalMatrix(tf.keras.layers.Layer):
    def __init__(self, name, num_units, **kwargs):
        super(DiagonalMatrix, self).__init__(name=name, **kwargs)
        self.num_units = num_units
        self.w = self.add_weight(
            name='w',
            shape=(num_units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi)
        )

    # z: [batch_sz, num_units]
    def mul(self, z): 
        vec = tf.complex(tf.cos(self.w), tf.sin(self.w))  # [num_units]
        return vec * z  # Broadcasting over batch dimension

# Reflection unitary matrix
class ReflectionMatrix(tf.keras.layers.Layer):
    def __init__(self, name, num_units, **kwargs):
        super(ReflectionMatrix, self).__init__(name=name, **kwargs)
        self.num_units = num_units
        self.re = self.add_weight(
            name="re",
            shape=(num_units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        )
        self.im = self.add_weight(
            name="im",
            shape=(num_units,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        )

    # z: [batch_sz, num_units]
    def mul(self, z):
        v = tf.complex(self.re, self.im)  # [num_units]
        vstar = tf.math.conj(v)  # [num_units]
        v = tf.expand_dims(v, 1)  # [num_units, 1]
        vstar = tf.expand_dims(vstar, 1)  # [num_units, 1]
        vstar_z = tf.matmul(z, vstar)  # [batch_size, 1]
        sq_norm = tf.reduce_sum(tf.abs(v[:, 0])**2)  # [1]
        factor = (2 / tf.complex(sq_norm, 0.0))
        return z - factor * tf.matmul(vstar_z, tf.transpose(v))

# Permutation unitary matrix
class PermutationMatrix:
    def __init__(self, name, num_units):
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.P = tf.constant(perm, tf.int32)

    # z: [batch_sz, num_units], permute columns
    def mul(self, z): 
        return tf.transpose(tf.gather(tf.transpose(z), self.P))

# FFTs
# z: complex[batch_sz, num_units]

def FFT(z):
    return tf.signal.fft(z) 

def IFFT(z):
    return tf.signal.ifft(z) 
    
def normalize(z):
    norm = tf.sqrt(tf.reduce_sum(tf.abs(z)**2))
    factor = (norm + 1e-6)
    return tf.complex(tf.math.real(z) / factor, tf.math.imag(z) / factor)

# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z, bias):  # relu(|z|+b) * (z / |z|)
    norm = tf.abs(z)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    scaled = tf.complex(tf.math.real(z) * scale, tf.math.imag(z) * scale)
    return scaled

###################################################################################################

# URNNCell implementation
class URNNCell(tf.keras.layers.Layer):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the URNN cell, hidden layer size.
        num_in (int): Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in, **kwargs):
        super(URNNCell, self).__init__(**kwargs)
        self.num_units = num_units
        self.num_in = num_in

        # set up input -> hidden connection
        self.w_ih = self.add_weight(
            shape=(2 * num_units, num_in),
            initializer=tf.keras.initializers.GlorotUniform(),
            name="w_ih"
        )
        self.b_h = self.add_weight(
            shape=(num_units,),
            initializer='zeros',
            name="b_h"
        )

        # Elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)
        self.P = PermutationMatrix("P", num_units)

    @property
    def state_size(self):
        return self.num_units * 2  # Real-valued state size

    @property
    def output_size(self):
        return self.num_units * 2  # Real-valued output size

    def call(self, inputs, states):
        """The most basic URNN cell.
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, num_in].
            states (list of Tensor): Previous state tensor of shape [batch_size, 2 * num_units].
        Returns:
            output (Tensor): Output tensor of shape [batch_size, 2 * num_units].
            new_state (list of Tensor): New state tensor of shape [batch_size, 2 * num_units].
        """
        # Previous state
        state = states[0]  # [batch_size, 2 * num_units]

        # Prepare input linear combination
        inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih))  # [batch_size, 2 * num_units]
        inputs_mul_c = tf.complex(
            inputs_mul[:, :self.num_units], 
            inputs_mul[:, self.num_units:]
        )  # [batch_size, num_units]

        # Prepare state linear combination (always complex)
        state_c = tf.complex(
            state[:, :self.num_units], 
            state[:, self.num_units:]
        )  # [batch_size, num_units]

        # Apply unitary transformations
        state_mul = self.D1.mul(state_c)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
        state_mul = self.D2.mul(state_mul)
        state_mul = IFFT(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul)  # [batch_size, num_units]

        # Calculate preactivation
        preact = inputs_mul_c + state_mul  # [batch_size, num_units]

        # Apply modReLU activation
        new_state_c = modReLU(preact, self.b_h)  # [batch_size, num_units] (complex)
        new_state = tf.concat(
            [tf.math.real(new_state_c), tf.math.imag(new_state_c)], axis=1
        )  # [batch_size, 2 * num_units] (real)

        # Output is the new state
        output = new_state
        return output, [new_state]