import tensorflow as tf
from TacotronModel import Encoder

class Postnet():
    """Postnet that takes final decoder output and fine tunes it (using vision on past and future frames)
	"""

    def __init__(self, is_training, hparams, activation=tf.nn.tanh, scope=None):
        """
		Args:
			is_training: Boolean, determines if the model is training or in inference to control dropout
			kernel_size: tuple or integer, The size of convolution kernels
			channels: integer, number of convolutional kernels
			activation: callable, postnet activation function for each convolutional layer
			scope: Postnet scope.
		"""
        super(Postnet, self).__init__()
        self.is_training = is_training

        self.kernel_size = hparams.postnet_kernel_size
        self.channels = hparams.postnet_channels
        self.activation = activation
        self.scope = 'postnet_convolutions' if scope is None else scope
        self.postnet_num_layers = hparams.postnet_num_layers
        self.drop_rate = hparams.dropout_rate

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            x = inputs
            for i in range(self.postnet_num_layers - 1):
                x = Encoder.conv1d(x, self.kernel_size, self.channels, self.activation,
                           self.is_training, self.drop_rate, 'conv_layer_{}_'.format(i + 1) + self.scope)
            x = Encoder.conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate,
                       'conv_layer_{}_'.format(5) + self.scope)
        return x

