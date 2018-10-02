from TacotronModel import Encoder, Decoder, Postnet
import tensorflow as tf
from Utils.Helpers import TacoTestHelper,TacoTrainingHelper
from tensorflow.contrib.seq2seq import dynamic_decode
from Utils.Infolog import log
from Utils.Utils import MaskedMSE, MaskedSigmoidCrossEntropy
from Utils.TextProcessing.HangulUtils import hangul_symbol_1,hangul_symbol_2,hangul_symbol_3,hangul_symbol_4,hangul_symbol_5


class Tacotron():
    '''Tacotron model, the wrapper of Encoder and Decoder model
    :arg
        hparams: hyper parameters to use'''

    def __init__(self, hparams):
        self.hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None,
                   targets_lengths=None, GTA=False, global_step=None, is_training=False, is_evaluating=False):
        """
        		Initializes the model for inference

        		sets "mel_outputs" and "alignments" fields.

        		Args:
        			- inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        			  steps in the input time series, and values are character IDs
        			- input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        			of each sequence in inputs.
        			- mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        			of steps in the output time series, M is num_mels, and values are entries in the mel
        			spectrogram. Only needed for training.
        		"""

        ### checking for conditions
        if mel_targets is None and stop_token_targets is not None:
            raise ValueError('no mel targets were provided but token_targets were given')
        if mel_targets is not None and stop_token_targets is None and not GTA:
            raise ValueError('Mel targets are provided without corresponding token_targets')
        if not GTA and self.hparams.predict_linear == True and linear_targets is None and is_training:
            raise ValueError(
                'Model is set to use post processing to predict linear spectrograms in training but no linear targets given!')
        if GTA and linear_targets is not None:
            raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
        if is_training and self.hparams.mask_decoder and targets_lengths is None:
            raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
        if is_training and is_evaluating:
            raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

        ####### declare variables
        with tf.variable_scope('inference') as scope:
            batch_size = tf.shape(inputs)[0]
            hparams = self.hparams
            assert hparams.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
            if hparams.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
                assert global_step is not None


            ### get symbol to create embedding lookup table
            if self.hparams.hangul_type == 1:
                hangul_symbol = hangul_symbol_1
            elif self.hparams.hangul_type == 2:
                hangul_symbol = hangul_symbol_2
            elif self.hparams.hangul_type == 3:
                hangul_symbol = hangul_symbol_3
            elif self.hparams.hangul_type == 4:
                hangul_symbol = hangul_symbol_4
            else:
                hangul_symbol = hangul_symbol_5

            # Embeddings ==> [batch_size, sequence_length, embedding_dim]
            # create embedding look up table with shape of [number of symbols, embedding dimension (declare in hparams]
            embedding_table = tf.get_variable(
                'inputs_embedding', [len(hangul_symbol), hparams.embedding_dim], dtype=tf.float32)
            ### inputs is a tensor of sequence of IDs  (created using text_to_sequence)
            # which is loaded through feeder class (_meta_data variable) from train.txt in training_data folder
            ## embedded_input is a Tensor with same type with embedding_table Tensor
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)
            ##########################
            ## create Encoder object##
            ##########################
            encoder_cell = Encoder.EncoderCell(
                EncoderConvolution=Encoder.EncoderConvolution(
                    is_training = is_training,
                    hparams = hparams,
                    scope='encoder_convolutions'
                ),
                EncoderLSTM = Encoder.EncoderLSTM(
                    is_training= is_training,
                    size=256,
                    zoneout=0.1,
                    scope='encoder_lstm'
                )
            )
            # extract Encoder model output
            encoder_outputs = encoder_cell(embedded_inputs, input_lengths=input_lengths)

            # store convolution output shape for visualization
            enc_conv_output_shape = encoder_cell.conv_output_shape

            ##########################
            ## create Decoder object##
            ##########################
            decoder_cell = Decoder.TacotronDecoderCell(
                prenet = Decoder.Prenet(
                    is_training=is_training,
                    layers_sizes=[256, 256],
                    drop_rate=hparams.dropout_rate,
                    scope='decoder_prenet'
                ),
                attention_mechanism = Decoder.LocationSensitiveAttention(
                    num_units=hparams.attention_dim,
                    memory=encoder_outputs,
                    hparams=hparams,
                    mask_encoder=True,
                    memory_sequence_length=input_lengths,
                    cumulate_weights=True
                ),
                rnn_cell = Decoder.DecoderRNN(
                    is_training=is_training,
                    layers=2,
                    zoneout=hparams.zoneout_rate,
                    scope='decoder_lstm'
                ),
                frame_projection = Decoder.FrameProjection(
                    shape=hparams.num_mels*2,
                    scope='linear_transform'
                ),
                stop_projection = Decoder.StopProjection(
                    is_training=is_training or is_evaluating,
                    shape=2,
                    scope='stop_token_projection'
                ),
            )
            ##initiate the first state of decoder
            decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            # Define the helper for our decoder
            if is_training or is_evaluating or GTA:
                self.helper = TacoTrainingHelper(batch_size, mel_targets, stop_token_targets, hparams, GTA, is_evaluating,
                                                 global_step)
            else:
                self.helper = TacoTestHelper(batch_size, hparams)

            # Decode
            (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
                Decoder.CustomDecoder(decoder_cell, self.helper, decoder_init_state),
                impute_finished=False,
                maximum_iterations=hparams.max_iters,
                swap_memory=hparams.tacotron_swap_with_cpu)







            # Only use max iterations at synthesis time
            max_iters = hparams.max_iters if not (is_training or is_evaluating) else None
            # Reshape outputs to be one output per entry
            # ==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
            decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hparams.num_mels])
            stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

            # Postnet
            postnet = Postnet.Postnet(is_training, hparams=hparams, scope='postnet_convolutions')

            # Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
            residual = postnet(decoder_output)

            # Project residual to same dimension as mel spectrogram
            # ==> [batch_size, decoder_steps * r, num_mels]
            residual_projection = Decoder.FrameProjection(hparams.num_mels, scope='postnet_projection')
            projected_residual = residual_projection(residual)

            # Compute the mel spectrogram
            mel_outputs = decoder_output + projected_residual

            # time-domain waveforms is only used for predicting mels to train wavenet vocoder\
            # so we omit post processing when doing GTA synthesis
            post_condition = hparams.predict_linear and not GTA
            if post_condition:
                # Based on https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                # Post-processing Network to map mels to linear spectrograms using same architecture as the encoder
                post_processing_cell = Encoder.EncoderCell(
                    Encoder.EncoderConvolution(is_training, hparams=hparams, scope='post_processing_convolutions'),
                    Encoder.EncoderLSTM(is_training, size=hparams.enc_lstm_hidden_size,
                               zoneout=hparams.zoneout_rate, scope='post_processing_LSTM'))

                expand_outputs = post_processing_cell(mel_outputs)
                linear_outputs = Decoder.FrameProjection(hparams.num_freq, scope='post_processing_projection')(expand_outputs)
                self.linear_outputs = linear_outputs
                self.linear_targets = linear_targets
            # Grab alignments from the final decoder state
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

            if is_training:
                self.ratio = self.helper._ratio
            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_output = decoder_output
            self.alignments = alignments
            self.stop_token_prediction = stop_token_prediction
            self.stop_token_targets = stop_token_targets
            self.mel_outputs = mel_outputs

            self.mel_targets = mel_targets
            self.targets_lengths = targets_lengths
            log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
            log('  Train mode:               {}'.format(is_training))
            log('  Eval mode:                {}'.format(is_evaluating))
            log('  GTA mode:                 {}'.format(GTA))
            log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
            log('  embedding:                {}'.format(embedded_inputs.shape))
            log('  enc conv out:             {}'.format(enc_conv_output_shape))
            log('  encoder out:              {}'.format(encoder_outputs.shape))
            log('  decoder out:              {}'.format(decoder_output.shape))
            log('  residual out:             {}'.format(residual.shape))
            log('  projected residual out:   {}'.format(projected_residual.shape))
            log('  mel out:                  {}'.format(mel_outputs.shape))
            if post_condition:
                log('  linear out:               {}'.format(linear_outputs.shape))
            log('  <stop_token> out:         {}'.format(stop_token_prediction.shape))

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self.hparams

            if hp.mask_decoder:
                # Compute loss of predictions before postnet
                before = MaskedMSE(self.mel_targets, self.decoder_output, self.targets_lengths,
                                   hparams=self.hparams)
                # Compute loss after postnet
                after = MaskedMSE(self.mel_targets, self.mel_outputs, self.targets_lengths,
                                  hparams=self.hparams)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = MaskedSigmoidCrossEntropy(self.stop_token_targets,
                                                            self.stop_token_prediction, self.targets_lengths,
                                                            hparams=self.hparams)
            else:
                # Compute loss of predictions before postnet
                before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
                # Compute loss after postnet
                after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
                # Compute <stop_token> loss (for learning dynamic generation stop)
                stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.stop_token_targets,
                    logits=self.stop_token_prediction))

            if hp.predict_linear:
                # Compute linear loss
                # From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
                # Prioritize loss for frequencies under 2000 Hz.
                l1 = tf.abs(self.linear_targets - self.linear_outputs)
                n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_mels)
                linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
            else:
                linear_loss = 0.

            # Compute the regularization weight
            if hp.tacotron_scale_regularization:
                reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
                reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
            else:
                reg_weight = hp.tacotron_reg_weight

            # Get all trainable variables
            all_vars = tf.trainable_variables()
            regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
                                       if not ('bias' in v.name or 'Bias' in v.name)]) * reg_weight

            # Compute final loss term
            self.before_loss = before
            self.after_loss = after
            self.stop_token_loss = stop_token_loss
            self.regularization_loss = regularization
            self.linear_loss = linear_loss
            self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.regularization_loss + self.linear_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self.hparams
            if hp.tacotron_decay_learning_rate:
                self.decay_steps = hp.tacotron_decay_steps
                self.decay_rate = hp.tacotron_decay_rate
                self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
                                               hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            # Just for causion
            # https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self.hparams
        # Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
                                        global_step - hp.tacotron_start_decay,  # lr = 1e-3 at step 50k
                                        self.decay_steps,
                                        self.decay_rate,  # lr = 1e-5 around step 310k
                                        name='lr_exponential_decay')

        # clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)