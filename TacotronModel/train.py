import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' ### disable warning messages
import subprocess
import time
import tensorflow as tf
import traceback
import argparse
from Utils.Hyperparams import hparams
from Utils import Infolog
from Utils.TextProcessing.TextPreprocessing import sequence_to_text
from Utils.AudioProcessing.AudioPreprocess import *
from Utils.Feeder import Feeder
from TacotronModel.Tacotron import Tacotron
from Utils.Plot import plot_alignment, plot_spectrogram
from Utils.Utils import ValueWindow
log = Infolog.log








def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, 'taco_pretrained/')
    checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
    input_path = os.path.join(args.base_dir, args.tacotron_input)
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    mel_dir = os.path.join(log_dir, 'mel-spectrograms')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    ## check whethe post-processing network will be used for linear spectrogram prediction
    if hparams.predict_linear:
        linear_dir = os.path.join(log_dir, 'linear-spectrograms')
        os.makedirs(linear_dir, exist_ok=True)
    # save log info
    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))
    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tacotron_random_seed)
    # Set up data feeder
    ## create an object of Feeder class to feed preprocessed data:
    #  (audio time series, mel spectrogram matrix, text sequences) to training model
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = Feeder(coord, input_path, hparams)
########################################
    # Set up model:
    # create model based on '--model' parameters ('Tacotron', 'Tacotron2', 'WaveNet', 'Both')
    global_step = tf.Variable(0, name='global_step', trainable=False) ## define global step to use in tf.train.cosine_decay() when using teacher forcing
    model, stats = model_train_mode(args, feeder, hparams, global_step)

    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5)
    log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))

    # Memory allocation on the GPU as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Train
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, Default = True.
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                except tf.errors.OutOfRangeError as e:
                    log('Cannot restore checkpoint: {}'.format(e))

            if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                saver.restore(sess, checkpoint_state.model_checkpoint_path)

            else:
                if not args.restore:
                    log('Starting new training!')
                else:
                    log('No model to load at {}'.format(save_dir))

            # initializing feeder
            ## feed preprocessed data to threads
            feeder.start_threads(sess)
            # Training loop
            while not coord.should_stop() and step < args.tacotron_train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                    step, time_window.average, loss, loss_window.average)
                log(message, end='\r')
                if np.isnan(loss):
                    log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                    raise Exception('Loss exploded')

                if step % args.summary_interval == 0:
                    log('\nWriting summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(stats), step)


                ##### save check point when check point interval has been met
                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps:
                    # Save model and current global step
                    saver.save(sess, checkpoint_path, global_step=global_step)

                    log('\nSaving Mel-Spectrograms and griffin-lim inverted waveform..')
                    if hparams.predict_linear:
                        input_seq, mel_prediction, linear_prediction, alignment, target, target_length = sess.run([
                            model.inputs[0],
                            model.mel_outputs[0],
                            model.linear_outputs[0],
                            model.alignments[0],
                            model.mel_targets[0],
                            model.targets_lengths[0],
                        ])

                        # save predicted linear spectrogram to disk (debug)
                        linear_filename = 'linear-prediction-step-{}.npy'.format(step)
                        np.save(os.path.join(linear_dir, linear_filename), linear_prediction.T, allow_pickle=False)

                        # save griffin lim inverted wav for debug (linear -> wav)
                        wav = linear_to_audio_serie(linear_prediction.T, hparams)
                        save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-linear.wav'.format(step)),
                                       sr=hparams.sample_rate)

                    else:
                        input_seq, mel_prediction, alignment, target, target_length = sess.run([model.inputs[0],
                                                                                                model.mel_outputs[0],
                                                                                                model.alignments[0],
                                                                                                model.mel_targets[0],
                                                                                                model.targets_lengths[0]
                                                                                                ])
                    # save predicted mel spectrogram to disk (debug)
                    mel_filename = 'mel-prediction-step-{}.npy'.format(step)
                    np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)

                    # save griffin lim inverted wav for debug (mel -> wav)
                    wav = mel_to_audio_serie(mel_prediction.T, hparams)
                    save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)),
                                   sr=hparams.sample_rate)

                    # save alignment plot to disk (control purposes)
                    plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
                                        info='{}, {}, step={}, loss={:.5f}'.format(args.model, datetime.now().strftime('%Y-%m-%d %H:%M'), step,
                                                                                   loss),
                                        max_len=target_length // hparams.outputs_per_step)
                    # save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot_spectrogram(mel_prediction,
                                          os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
                                          info='{}, {}, step={}, loss={:.5}'.format(args.model, datetime.now().strftime('%Y-%m-%d %H:%M'), step,
                                                                                    loss), target_spectrogram=target,
                                          max_len=target_length)
                    log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))
                    ##### FINISH training
                    ##### Testing....
                    ## do the test when step is the maximum of tacotron training step
            return save_dir
        except Exception as e:
            log('Exiting due to exception: {}'.format(e))
            traceback.print_exc()
            coord.request_stop(e)

def add_train_stats(model, hparams):
    with tf.variable_scope('stats') as scope:
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)
        if hparams.predict_linear:
            tf.summary.scalar('linear_loss', model.linear_loss)
        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('stop_token_loss', model.stop_token_loss)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)  # Control learning rate decay speed
        if hparams.tacotron_teacher_forcing_mode == 'scheduled':
            tf.summary.scalar('teacher_forcing_ratio',
                              model.ratio)  # Control teacher forcing ratio decay when mode = 'scheduled'
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm',
                          tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
        return tf.summary.merge_all()

def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        #create Tacotron model object
        model = Tacotron(hparams)
        #initialize Tacotron model parameters based on type of predict (linear or not)
        if hparams.predict_linear:
            model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
                             linear_targets=feeder.linear_targets,
                             targets_lengths=feeder.targets_lengths, global_step=global_step,
                             is_training=True)
        else:
            model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
                             targets_lengths=feeder.targets_lengths, global_step=global_step,
                             is_training=True)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_train_stats(model, hparams)
        return model, stats


def model_test_mode(args, feeder, hparams, global_step):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        # create Tacotron model object
        model = Tacotron(hparams)
        # initialize Tacotron model parameters based on type of predict (linear or not)
        if hparams.predict_linear:
            model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets,
                             feeder.eval_token_targets,
                             linear_targets=feeder.eval_linear_targets, targets_lengths=feeder.eval_targets_lengths,
                             global_step=global_step,
                             is_training=False, is_evaluating=True)
        else:
            model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, feeder.eval_mel_targets,
                             feeder.eval_token_targets,
                             targets_lengths=feeder.eval_targets_lengths, global_step=global_step, is_training=False,
                             is_evaluating=True)
        model.add_loss()
        return model



def GetArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--tacotron_input', default='Tacotron_input/train.txt')
    parser.add_argument('--wavenet_input', default='tacotron_output/GTA/map.txt')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='synthesis', help='mode for synthesis of tacotron after training')
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in Tacotron synthesis mode')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help='Steps between eval on test data')
    parser.add_argument('--tacotron_train_steps', type=int, default=100000,
                        help='total number of tacotron training steps')
    parser.add_argument('--wavenet_train_steps', type=int, default=360000,
                        help='total number of wavenet training steps')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = GetArguments()
    train('Tacotron_trained_logs',args, hparams=hparams)
