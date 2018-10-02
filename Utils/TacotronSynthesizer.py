import os
import wave

import numpy as np
import pyaudio
import tensorflow as tf

from TacotronModel.Tacotron import Tacotron
from Utils.AudioProcessing.AudioPreprocess import mel_to_audio_serie, linear_to_audio_serie, save_wav
from Utils.Infolog import log
from Utils.Plot import plot_spectrogram, plot_alignment
from Utils.TextProcessing.HangulUtils import hangul_to_sequence


class Synthesizer:
    def load(self, checkpoint_path, hparams, GTA=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
        with tf.variable_scope('model') as scope:
            self.model = Tacotron(hparams)
            if GTA:
                self.model.initialize(inputs, input_lengths, targets, GTA=GTA)
            else:
                self.model.initialize(inputs, input_lengths)
            self.mel_outputs = self.model.mel_outputs
            self.alignment = self.model.alignments[0]

        self.gta = GTA
        self._hparams = hparams

        log('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text, index, out_dir, log_dir, mel_filename):
        hparams = self._hparams
        seq = hangul_to_sequence(dir=hparams.base_dir, hangul_text=text, hangul_type=hparams.hangul_type)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
        }

        if self.gta:
            feed_dict[self.model.mel_targets] = np.load(mel_filename).reshape(1, -1, hparams.num_mels)

        if self.gta or not hparams.predict_linear:
            mels, alignment = self.session.run([self.mel_outputs, self.alignment], feed_dict=feed_dict)

        else:
            linear, mels, alignment = self.session.run([self.linear_outputs, self.mel_outputs, self.alignment],
                                                       feed_dict=feed_dict)
            linear = linear.reshape(-1, hparams.num_freq)

        mels = mels.reshape(-1, hparams.num_mels)  # Thanks to @imdatsolak for pointing this out

        if index is None:
            # Generate wav and read it
            wav = mel_to_audio_serie(mels.T, hparams)
            save_wav(wav, 'temp.wav', sr=hparams.sample_rate)  # Find a better way

            chunk = 512
            f = wave.open('temp.wav', 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels=f.getnchannels(),
                            rate=f.getframerate(),
                            output=True)
            data = f.readframes(chunk)
            while data:
                stream.write(data)
                data = f.readframes(chunk)

            stream.stop_stream()
            stream.close()

            p.terminate()
            return

        # Write the spectrogram to disk
        # Note: outputs mel-spectrogram files and target ones have same names, just different folders
        mel_filename = os.path.join(out_dir, 'speech-mel-{:05d}.npy'.format(index))
        np.save(mel_filename, mels, allow_pickle=False)

        if log_dir is not None:
            # save wav (mel -> wav)
            wav = mel_to_audio_serie(mels.T, hparams)
            save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-mel.wav'.format(index)),
                     sr=hparams.sample_rate)

            if hparams.predict_linear:
                # save wav (linear -> wav)
                wav = linear_to_audio_serie(linear.T, hparams)
                save_wav(wav, os.path.join(log_dir, 'wavs/speech-wav-{:05d}-linear.wav'.format(index)),
                         sr=hparams.sample_rate)

            # save alignments
            plot_alignment(alignment, os.path.join(log_dir, 'plots/speech-alignment-{:05d}.png'.format(index)),
                           info='{}'.format(text), split_title=True)

            # save mel spectrogram plot
            plot_spectrogram(mels, os.path.join(log_dir, 'plots/speech-mel-{:05d}.png'.format(index)),
                             info='{}'.format(text), split_title=True)

        return mel_filename
