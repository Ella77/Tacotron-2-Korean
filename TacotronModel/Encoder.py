import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

class EncoderCell():
    """Encoder Cell
	Wrap TacotronEncoderConvoluton class and TacotronEncoderBidirectionalLSTM class to form Encoder Cell
	:arg
	    _convolutions: TacotronEncoderConvolution class object
	    _cell: TacotronEncoderBidirectionalLSTM class object
	"""
    def __init__(self, EncoderConvolution, EncoderLSTM):
        super(EncoderCell, self).__init__()
        self.convolution_network = EncoderConvolution
        self.lstm_network = EncoderLSTM

    def __call__(self, inputs, input_lengths = None):
        conv_output = self.convolution_network(inputs)
        lstm_output = self.lstm_network(conv_output, input_lengths)
        self.conv_output_shape = conv_output.shape
        return lstm_output

class EncoderConvolution():
    '''Declare convolution network class
    :arg:
        is_training: whether model is used for training
        kernel_sze: convolution filter size
        channels: number of kernels to be used
        activation: activation funtion to use
        scope: network scope
    '''
    def __init__(self, is_training, hparams, activation = tf.nn.relu, scope='encoder_convolutions'):
        self.is_training = is_training
        self.activation = activation
        self.scope = scope
        self.kernel_size = hparams.conv_kernel_size
        self.channels = hparams.conv_channels
        self.drop_rate = hparams.dropout_rate
        self.num_layers = hparams.conv_num_layers

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.num_layers):
                x = conv1d(x, self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate, 'conv_layer_{}_'.format(i + 1) + self.scope)
        return x

class EncoderLSTM():
    '''Declare bidirectional lstm network class
        :arg:
            is_training: whether model is used for training
            size: hidden units size
            zoneout: zoneout rate for zoneout lstm
            scope: network scope
        '''

    def __init__(self, is_training, size=256, zoneout=0.1, scope='encoder_lstm'):
        super(EncoderLSTM, self).__init__()
        self.is_training = is_training
        self.size = size
        self.zoneout_rate = zoneout
        self.scope = scope
        self.fw_cell = ZoneoutLSTMCell(size, is_training, zoneout_factor_output=zoneout, zoneout_factor_cell=zoneout, name='encoder_fw_lstm')
        self.bw_cell = ZoneoutLSTMCell(size, is_training, zoneout_factor_output=zoneout, zoneout_factor_cell=zoneout, name='encoder_bw_lstm')


    def __call__(self, inputs, input_lengths):
        with tf.variable_scope(self.scope):
            outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell,
                self.bw_cell,
                inputs,
                sequence_length=input_lengths,
                dtype=tf.float32,
                swap_memory=True)
            return tf.concat(outputs, axis=2)  # Concat and return forward + backward outputs

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper for tf LSTM to create Zoneout LSTM Cell
	inspired by:
	https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
	Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.
	Many thanks to @Ondal90 for pointing this out. You sir are a hero!

	Create zoneoutLSTMcell from original LSTMCell by applying zoneout to output and state
	This Cell will be used to create LSTMCell in both Encoder and Decoder model
	:arg:
	    num_units: LSTM hidden units size
	    is_training: whether moder is using for training
	    zoneout_factor_cell, zoneout_factor_output: zoneout's factor parameters
	    state_is_tuple: boolean, output state is an iterable object
	"""

    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True,
                 name=None):
        super(ZoneoutLSTMCell, self).__init__()
        # check for zoneout factor requirement (between [0,1]
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)
        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        # Apply zoneout
        if self.is_training:
            # nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h
        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])
        return output, new_state


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope):
    '''function to create tensorflow 1-D convolutional layer
    input:
        inputs: input vectors
        kernel_size: convolutional layer filter size ( , )
        channel: convolution channels
        activation: activiation function to use
        is_raining: boolean, whether model is being used for training
        drop_rate: convolutional network droprate
        scope: network scope name
    output:
        output tensor after dropped out
    '''
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs=inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=None,
            padding='same')
        batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
        activated = activation(batched)
        return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
                                 name='dropout_{}'.format(scope))
