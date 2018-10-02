import os
import argparse
import time

import tensorflow as tf
from tqdm import tqdm
from Utils.Infolog import log
from Utils.TacotronSynthesizer import Synthesizer
from Utils.Hyperparams import hparams

def run_synthesis(args, checkpoint_path, output_dir, hparams):
    GTA = (args.GTA == 'True')
    if GTA:
        synth_dir = os.path.join(output_dir, 'gta')
        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)
    else:
        synth_dir = os.path.join(output_dir, 'natural')
        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.input_dir, 'train.txt')
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, GTA=GTA)
    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        frame_shift_ms = hparams.hop_size / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / (3600)
        log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))
    time.sleep(1)
    log('starting synthesis..')
    mel_dir = os.path.join(args.input_dir, 'mels')
    wav_dir = os.path.join(args.input_dir, 'audio')
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            text = meta[5]
            mel_filename = os.path.join(mel_dir, meta[1])
            wav_filename = os.path.join(wav_dir, meta[0])
            mel_output_filename = synth.synthesize(text, i + 1, synth_dir, None, mel_filename)

            file.write('{}|{}|{}|{}\n'.format(wav_filename, mel_filename, mel_output_filename, text))
    print('done!')
    time.sleep(1)
    print('Predicted mel spectrograms are saved in {}'.format(synth_dir))
    print('Exitting...')
    time.sleep(3)
    return os.path.join(synth_dir, 'map.txt')

def run_inference(checkpoint_path, output_dir, hparams, sentences):
    print('creating folders for inference..')
    inference_dir = os.path.join(output_dir, 'inference')
    log_dir = os.path.join(output_dir, 'logs-inference')
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    time.sleep(1)
    print('done!')
    time.sleep(1)
    print('running inference..')
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    with open(os.path.join(inference_dir, 'map.txt'), 'w') as file:
        for i, text in enumerate(tqdm(sentences)):
            mel_filename = synth.synthesize(text, i + 1, inference_dir, log_dir, None)

            file.write('{}|{}\n'.format(text, mel_filename))
    log('synthesized mel spectrograms of \"{}\" at {}'.format(sentences,inference_dir))
    return inference_dir

def tacotron_synthesize(args, hparams, checkpoint):
    output_dir = 'tacotron_' + args.output_dir

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except AttributeError:
        # Swap logs dir name in case user used Tacotron-2 for train and Both for test (and vice versa)
        if 'Both' in checkpoint:
            checkpoint = checkpoint.replace('Both', 'Tacotron-2')
        elif 'Tacotron-2' in checkpoint:
            checkpoint = checkpoint.replace('Tacotron-2', 'Both')
        else:
            raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(checkpoint))
        try:
            # Try loading again
            checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
            log('loaded model at {}'.format(checkpoint_path))
        except:
            raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))
    return run_synthesis(args, checkpoint_path, output_dir, hparams)



def tacotron_inference(args, hparams, checkpoint, sentences):
    output_dir = 'tacotron_' + args.output_dir
    if sentences is None:
        raise RuntimeError('Inference mode requires input sentence(s), make sure you put sentences in sentences.txt!')

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except AttributeError:
        # Swap logs dir name in case user used Tacotron-2 for train and Both for test (and vice versa)
        if 'Both' in checkpoint:
            checkpoint = checkpoint.replace('Both', 'Tacotron-2')
        elif 'Tacotron-2' in checkpoint:
            checkpoint = checkpoint.replace('Tacotron-2', 'Both')
        else:
            raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(checkpoint))
        try:
            # Try loading again
            checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
            log('loaded model at {}'.format(checkpoint_path))
        except:
            raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))
    return run_inference(checkpoint_path, output_dir, hparams, sentences)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='Tacotron_trained_logs/taco_pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--mode', default='synthesize', help='runing mode, could be synthesize or inference')
    parser.add_argument('--input_dir', default='Tacotron_input/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=get_arguments()
    if args.mode =='synthesize':
        out_dir = tacotron_synthesize(args, hparams, args.checkpoint)
    elif args.mode =='inference':
        sentences = []
        with open(os.path.join('', 'sentences.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                sentences.append(line)
        inference_dir = tacotron_inference(args,hparams,args.checkpoint,hparams.sentences)
