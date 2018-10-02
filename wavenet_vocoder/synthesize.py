import argparse
import os

import numpy as np
import tensorflow as tf
from Utils.Infolog import log
from tqdm import tqdm
from wavenet_vocoder.synthesizer import Synthesizer
from Utils.Hyperparams import hparams


# def run_synthesis(args, checkpoint_path, output_dir, hparams):
#     log_dir = os.path.join(output_dir, 'plots')
#     wav_dir = os.path.join(output_dir, 'wavs')
#     synth = Synthesizer()
#     synth.load(checkpoint_path, hparams)
#     if args.model == 'Tacotron':
#         raise RuntimeError('Please run Tacotron synthesis from Tacotron folder, not here..')
#     else:
#         ### get mel file (inference result from tacotron model) from input mels_dir
#         mel_files = [os.path.join(args.mels_dir, f) for f in os.listdir(args.mels_dir) if f.split('.')[-1] == 'npy']
#         texts = None
#         ### create result folders
#         os.makedirs(log_dir, exist_ok=True)
#         os.makedirs(wav_dir, exist_ok=True)
#         log('Starting wavenet synthesis! (this will take a while..)')
#         ### split mels file (numpy matrix) to small parts due to wavenet_synthesis_batch_size in hyperparams
#         mel_files = [mel_files[i: i + hparams.wavenet_synthesis_batch_size] for i in
#                      range(0, len(mel_files), hparams.wavenet_synthesis_batch_size)]
#         ### open map.txt file and write down result
#         with open(os.path.join(wav_dir, 'map.txt'), 'w') as file:
#             #### loop over mel_files (remember that at here, mel_files are an numpy matrix)
#             for i, mel_batch in enumerate(tqdm(mel_files)):
#                 #### load numpy matrix of mel file
#                 mel_spectros=[np.load(mel) for mel in mel_batch]
#                 ### get npy file name
#                 basenames = [os.path.basename(mel).replace('.npy','') for mel in mel_batch]
#                 ### generate audio and save in wav_dir
#                 audio_files = synth.synthesize(mel_spectros, None, basenames, wav_dir, log_dir)
#
#                 speaker_logs = ['<no_g>'] * len(mel_batch)
#                 ###write down result
#                 for j, mel_file in enumerate(mel_batch):
#                     if texts is None:
#                         file.write('{}|{}\n'.format(mel_file, audio_files[j], speaker_logs[j]))
#                     else:
#                         file.write('{}|{}|{}\n'.format(texts[i][j], mel_file, audio_files[j], speaker_logs[j]))
#         log('synthesized audio waveforms at {}'.format(wav_dir))
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
    # if args.model == 'Tacotron-2':
    #     # If running all Tacotron-2, synthesize audio from evaluated mels
    #     metadata_filename = os.path.join(args.mels_dir, 'map.txt')
    #     with open(metadata_filename, encoding='utf-8') as f:
    #         metadata = np.array([line.strip().split('|') for line in f])
    #
    #     speaker_ids = metadata[:, 2]
    #     mel_files = metadata[:, 1]
    #     texts = metadata[:, 0]
    #
    #     speaker_ids = None if (speaker_ids == '<no_g>').all() else speaker_ids
    # else:
    #     # else Get all npy files in input_dir (supposing they are mels)
    #     mel_files = [os.path.join(args.mels_dir, f) for f in os.listdir(args.mels_dir) if f.split('.')[-1] == 'npy']
    #     speaker_ids = None if args.speaker_id is None else args.speaker_id.replace(' ', '').split(',')
    #
    #     if speaker_ids is not None:
    #         assert len(speaker_ids) == len(mel_files)
    #
    #     texts = None
    #
    # log('Starting synthesis! (this will take a while..)')
    # os.makedirs(log_dir, exist_ok=True)
    # os.makedirs(wav_dir, exist_ok=True)
    #
    # mel_files = [mel_files[i: i + hparams.wavenet_synthesis_batch_size] for i in
    #              range(0, len(mel_files), hparams.wavenet_synthesis_batch_size)]
    # texts = None if texts is None else [texts[i: i + hparams.wavenet_synthesis_batch_size] for i in
    #                                     range(0, len(texts), hparams.wavenet_synthesis_batch_size)]
    #
    # with open(os.path.join(wav_dir, 'map.txt'), 'w') as file:
    #     for i, mel_batch in enumerate(tqdm(mel_files)):
    #         mel_spectros = [np.load(mel_file) for mel_file in mel_batch]
    #
    #         basenames = [os.path.basename(mel_file).replace('.npy', '') for mel_file in mel_batch]
    #         speaker_id_batch = None if speaker_ids is None else speaker_ids[i]
    #         audio_files = synth.synthesize(mel_spectros, speaker_id_batch, basenames, wav_dir, log_dir)
    #
    #         speaker_logs = ['<no_g>'] * len(mel_batch) if speaker_id_batch is None else speaker_id_batch
    #
    #         for j, mel_file in enumerate(mel_batch):
    #             if texts is None:
    #                 file.write('{}|{}\n'.format(mel_file, audio_files[j], speaker_logs[j]))
    #             else:
    #                 file.write('{}|{}|{}\n'.format(texts[i][j], mel_file, audio_files[j], speaker_logs[j]))
    #
    # log('synthesized audio waveforms at {}'.format(wav_dir))


def wavenet_synthesize(args, hparams, checkpoint):
    output_dir = 'wavenet_' + args.output_dir
    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))
    log_dir = os.path.join(output_dir, 'plots')
    wav_dir = os.path.join(output_dir, 'wavs')
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)
    if args.model == 'Tacotron':
        raise RuntimeError('Please run Tacotron synthesis from Tacotron folder, not here..')
    else:
        ### get mel file (inference result from tacotron model) from input mels_dir
        mel_files = [os.path.join(args.mels_dir, f) for f in os.listdir(args.mels_dir) if f.split('.')[-1] == 'npy']
        texts = None
        ### create result folders
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(wav_dir, exist_ok=True)
        log('Starting wavenet synthesis! (this will take a while..)')
        ### split mels file (numpy matrix) to small parts due to wavenet_synthesis_batch_size in hyperparams
        mel_files = [mel_files[i: i + hparams.wavenet_synthesis_batch_size] for i in
                     range(0, len(mel_files), hparams.wavenet_synthesis_batch_size)]
        log('debug: synthesize.py line 128')
        ### open map.txt file and write down result
        ii=0
        iii=0
        with open(os.path.join(wav_dir, 'map.txt'), 'w') as file:
            log('debug: synthesize.py line 131')
            #### loop over mel_files (remember that at here, mel_files are an numpy matrix)
            for i, mel_batch in enumerate(tqdm(mel_files)):
                log('debug: synthesize.py vong lap ben ngoai, ii={}'.format(ii))
                ii = ii + 1
                #### load numpy matrix of mel file
                mel_spectros = [np.load(mel) for mel in mel_batch]
                log('debug: synthesize.py line 136, mel_spectros={}'.format(mel_spectros))
                ### get npy file name
                basenames = [os.path.basename(mel).replace('.npy', '') for mel in mel_batch]
                log('debug: synthesize.py line 139, basenames={}'.format(basenames))
                ### generate audio and save in wav_dir
                audio_files = synth.synthesize(mel_spectros, None, basenames, wav_dir, log_dir)
                log('debug: synthesize.py line 142, audio_files={}'.format(audio_files))
                speaker_logs = ['<no_g>'] * len(mel_batch)
                log('debug: synthesize.py line 144, audio_files={}'.format(speaker_logs))
                ###write down result

                for j, mel_file in enumerate(mel_batch):
                    if texts is None:
                        file.write('{}|{}\n'.format(mel_file, audio_files[j], speaker_logs[j]))
                        log('debug: synthesize.py vong lap ben trong, iii={}'.format(iii))
                        iii = iii + 1
                    else:
                        file.write('{}|{}|{}\n'.format(texts[i][j], mel_file, audio_files[j], speaker_logs[j]))
                        log('debug: synthesize.py vong lap ben trong, iii={}'.format(iii))
                        iii = iii + 1
        log('synthesized audio waveforms at {}'.format(wav_dir))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='Wavenet_trained_logs/wave_pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--mels_dir', default='tacotron_output/inference/',
                        help='folder to contain mels to synthesize audio from using the Wavenet')
    parser.add_argument('--mode', default='inference', help='runing mode, could be synthesize or inference')
    parser.add_argument('--input_dir', default='Tacotron_input/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    wave_checkpoint = os.path.join(hparams.base_dir, args.checkpoint)
    if args.mode !='inference':
        raise RuntimeError('Since the inference take time, wavenet only runs in inference mode!\n'
                           'Please pass \'--mode=inference\' to the synthesizing command, or just run as default!')
    else:
        wavenet_synthesize(args, hparams, wave_checkpoint)
