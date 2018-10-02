from pydub import AudioSegment
import os
import pandas as pd
import librosa
from Utils.Hyperparams import hparams
import soundfile
'''
This file do not necessary in training or synthesizing Tacotron and Wavenet. 
It is used to reformat new wav files in case of training other voice using pre trained model

For example, you have trained Tacotron and Wavenet using a A dataset, then you want to train with B dataset, 
but B dataset wav file has different sample rate from A dataset. This file is using to upsampling sample rate.
also, it provide a tool to trim out silence at beginning and end of wav file. 
'''

def convert_to_wav(input_audio_files, output_dir=None):
    for file in os.listdir(input_audio_files):
        if file.endswith('.flac') or file.endswith('.mp3'):
            sound, sample_rate = soundfile.read(input_audio_files+file,samplerate=None)
            file = file.split('.')[0]
            if output_dir is None:
                output_dir= input_audio_files+'converted\\'
            os.makedirs(output_dir, exist_ok=True)
            soundfile.write(output_dir + file + '.wav', sound, sample_rate)



def resample_wav(input_wav_files, output_dir=None, sample_rate=hparams.sample_rate):
    '''(upsampling)Convert audio wav to hparams.sample_rate
    :arg input_wav_files wavs folder
    :arg output_dir folder to save new wavs to
    :arg sample_rate desired sample rate
    :except permission Error exception'''
    ### read files from input folder

    for file in os.listdir(input_wav_files):
        if file.endswith('.wav') or file.endswith('.flac') or file.endswith('.mp3'):
            ## load wav file with new sample rate
            new_sound, sr = librosa.load(input_wav_files+file, sr=None)
            new_sound = librosa.resample(new_sound, sr, sample_rate)
            ### create output_dir if neccessary
            if output_dir is None:
                output_dir = os.path.join(input_wav_files,'resampled\\')
                os.makedirs(output_dir, exist_ok=True)
            try:
                ### write sound to file with new sample rate
                librosa.output.write_wav(output_dir+file, new_sound, sample_rate)
            except PermissionError:
                raise ('Could not write wav file, check destination folder permission..')


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms
input_wav_files = 'D:\\locs\data\\audio_data\\zeroth_korean_male_20_6minutes\\converted\\wavs_16000Hz\\'

def trimming(input_wav_files, output_dir=None, natural_silence_pad=200):
    '''
    triming out silences (keep a little silence to make sound more natural
    :param input_wav_files: folder of wav files
    :param output_dir:  folder to export new wavs
    :param natural_silence_pad: size of silence to keep
    :return:
    '''
    files = []## storing files name
    length = []## storing files' length
    for file in os.listdir(input_wav_files):
        if file.endswith('.wav'):
            files.append(file)## add file name to list
            sound = AudioSegment.from_file(input_wav_files + file, format="wav")
            start_trim = detect_leading_silence(sound)
            end_trim = detect_leading_silence(sound.reverse())
            duration = len(sound)

            if start_trim > natural_silence_pad:
                start_trim = start_trim - natural_silence_pad ### if silence part greater than narural silence pad, trim it
            else:
                start_trim=0                 ### if silence part smaller than narural silence pad, do not trim it
            if end_trim > natural_silence_pad:
                end_trim = end_trim - natural_silence_pad ### if silence part greater than narural silence pad, trim it
            else:
                end_trim=0                  ### if silence part smaller than narural silence pad, do not trim it
            trimmed_sound = sound[start_trim:duration - end_trim]

            length.append(len(trimmed_sound)/1000) ## calculate file length (in ms) and add to length list
            ## create output_dir if necessary
            if output_dir is None:
                output_dir = input_wav_files + 'trimmed\\'
            os.makedirs(output_dir, exist_ok=True)
            ### write down sound file info ( file length, file name) into a csv file
            info_df = pd.DataFrame(files)
            info_df = info_df.assign(duaration=length)
            info_df.to_csv(output_dir + 'trimmed_wav_info.csv')
            try:
                trimmed_sound.export(output_dir+file, format='wav')
            except PermissionError:
                raise ('Could not write wav file, check output folder permission..')
