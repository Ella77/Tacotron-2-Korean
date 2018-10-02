import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from Utils.Utils import mulaw_quantize, mulaw, is_mulaw, is_mulaw_quantize
from Utils.Hyperparams import hparams
import os

def load_wav(path, sr):
    '''Load audio files (which have sample rate [sr]) from [path] folder
    input:
        path: audio file path
        sr: audio file's sample rate (22050 Hz in case of ljspeech1.1
    output:
        numpy array presenation of audio'''
    return librosa.core.load(path, sr=sr)[0]




def audio_series_to_mel(hparams, audio_series):
    ## explicitly load mel spectrogram parameters from hparams (for easy understanding)
    n_fft=hparams.n_fft                                     #int
    use_lws = hparams.use_lws                               #boolean
    ref_level_db = hparams.ref_level_db                     #int
    wav = audio_series
    win_size = hparams.win_size
    #check using Local Weighted Sum condition
    if use_lws:
        import lws
        weighted_sum = lws.lws(n_fft, get_hop_size(hparams), fftsize=win_size, mode="speech")
        D = weighted_sum.stft(wav).T
    else:
        D = librosa.stft(y=wav, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=win_size)
    #require max voice frequency is less than a half of sample rate
    assert hparams.fmax <= hparams.sample_rate // 2
    #### transforming audio time series to mel
    _mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)
    linear_mel = np.dot(_mel_basis, np.abs(D))###
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    dp = 20 * np.log10(np.maximum(min_level, linear_mel))
    S = dp - ref_level_db
    #check for normalizing condition
    if hparams.signal_normalization:
        mel =  _normalize(S, hparams)
    else:
        mel = S
    return mel


def audio_series_to_linear(hparams, audio_series):
    ## explicitly load mel spectrogram parameters from hparams (for easy understanding)
    n_fft=hparams.n_fft                                     #int
    win_size=hparams.win_size                               #int
    use_lws = hparams.use_lws                               #boolean
    ref_level_db = hparams.ref_level_db                     #int
    wav = audio_series
    #check using Local Weighted Sum condition
    if use_lws:
        import lws
        weighted_sum = lws.lws(n_fft, get_hop_size(hparams), fftsize=win_size, mode="speech")
        D = weighted_sum.stft(wav).T
    else:
        D = librosa.stft(y=wav, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=win_size)
    #require max voice frequency is less than a half of sample rate
    assert hparams.fmax <= hparams.sample_rate // 2
    #### transforming audio time series to mel
    _mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)
    linear = np.abs(D)###
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    dp = 20 * np.log10(np.maximum(min_level, linear))
    S = dp - ref_level_db
    #check for normalizing condition
    if hparams.signal_normalization:
        linear =  _normalize(S, hparams)
    else:
        linear = S
    return linear


def linear_to_audio_serie(linear, hparams):
    ## explicitly load mel spectrogram parameters from hparams (for easy understanding)
    n_fft = hparams.n_fft  # int
    win_size = hparams.win_size  # int
    use_lws = hparams.use_lws  # boolean
    if hparams.signal_normalization:
        S = _denormalize(linear, hparams)
    else:
        S = linear

    dp = S + hparams.ref_level_db
    amp = np.power(10.0, dp * 0.05)
    D = amp
    if use_lws:
        import lws
        weighted_sum = lws.lws(n_fft, get_hop_size(hparams), fftsize=win_size, mode="speech")
        D = weighted_sum.run_lws(D.astype(np.float64).T ** hparams.power)
        audio_series = weighted_sum.istft(D).astype(np.float32)
        return audio_series
    else:
        D = D ** hparams.power
        angles = np.exp(2j * np.pi * np.random.rand(*D.shape))
        S_complex = np.abs(D).astype(np.complex)
        reverse_y = librosa.istft(S_complex * angles, hop_length=get_hop_size(hparams), win_length=win_size)
        for i in range(hparams.griffin_lim_iters):
            reverse_y = librosa.stft(y=reverse_y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=win_size)
            angles = np.exp(1j * np.angle(reverse_y))
            audio_series = librosa.istft(S_complex * angles, hop_length=get_hop_size(hparams), win_length=win_size)
        return audio_series

def mel_to_audio_serie(mel, hparams):
    ## explicitly load mel spectrogram parameters from hparams (for easy understanding)
    n_fft = hparams.n_fft  # int
    win_size = hparams.win_size  # int
    use_lws = hparams.use_lws  # boolean
    if hparams.signal_normalization:
        S = _denormalize(mel, hparams)
    else:
        S = mel
    dp = S + hparams.ref_level_db
    amp = np.power(10.0, dp * 0.05)
    _mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                                     fmin=hparams.fmin, fmax=hparams.fmax)
    _inv_mel_basis = np.linalg.pinv(_mel_basis)
    D = np.maximum(1e-10, np.dot(_inv_mel_basis, amp))

    if use_lws:
        import lws
        weighted_sum = lws.lws(n_fft, get_hop_size(hparams), fftsize=win_size, mode="speech")
        D = weighted_sum.run_lws(D.astype(np.float64).T ** hparams.power)
        audio_series = weighted_sum.istft(D).astype(np.float32)
        return audio_series
    else:
        D = D ** hparams.power
        angles = np.exp(2j * np.pi * np.random.rand(*D.shape))
        S_complex = np.abs(D).astype(np.complex)
        reverse_y = librosa.istft(S_complex * angles, hop_length=get_hop_size(hparams), win_length=win_size)
        for i in range(hparams.griffin_lim_iters):
            reverse_y = librosa.stft(y=reverse_y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=win_size)
            angles = np.exp(1j * np.angle(reverse_y))
            audio_series = librosa.istft(S_complex * angles, hop_length=get_hop_size(hparams), win_length=win_size)
        return audio_series

def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
    """
	Preprocesses a single utterance wav/text pair
    convert audio data to:
	        - audio time serie data form (numpy array)
	        - mel + linear spectrogram matrix (numpy matrix)
	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
    try:
        # Load the audio as numpy array
        wav = load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    # rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # check for M-AILABS extra silence specific
    if hparams.trim_silence:
        wav = trim_silence(wav, hparams)

    # check for Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = mulaw_quantize(wav, hparams.quantize_channels)
        # Trim silences
        start, end = start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]
        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    #check for mu_larw
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = mulaw(wav, hparams.quantize_channels)
        constant_values = mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:
        # [-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio_series_to_mel(hparams, wav).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    print('Debug: audiopreprocessing.py 214 mel_frames={}'.format(mel_frames))
    print('Debug: audiopreprocessing.py 214 mel_spectrogram.shape={}'.format(mel_spectrogram.shape))
    print('Debug: audiopreprocessing.py 214 max_mel_frames={}'.format(hparams.max_mel_frames))

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio_series_to_linear(hparams, wav).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]
    # sanity check
    assert linear_frames == mel_frames

    # Ensure time resolution adjustement between audio and mel-spectrogram
    fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
    l, r = pad_lr(wav, fft_size, get_hop_size(hparams))
    # Zero pad for quantized signal
    out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    assert len(out) >= mel_frames * get_hop_size(hparams)
    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * get_hop_size(hparams)]
    assert len(out) % get_hop_size(hparams) == 0
    time_steps = len(out)
    # Write the spectrogram and audio to disk
    audio_filename = 'speech-audio-{:05d}.npy'.format(index)
    mel_filename = 'speech-mel-{:05d}.npy'.format(index)
    linear_filename = 'speech-linear-{:05d}.npy'.format(index)
    np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example
    return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)


def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    """
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 0
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, 'transcript.txt'), encoding='utf-8') as f: ### read input audio's file name and text to f variable
            i=0
            for line in f:
                parts = line.strip().split('|')
                wav_name = parts[0].strip().split('/')[1]
                wav_path = os.path.join(input_dir, 'wavs', '{}'.format(wav_name))
                text = parts[2]
                futures.append(executor.submit(
                    partial(_process_utterance, mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams)))
                index +=1
                i += 1
                print('debug: AudioPreprocessing.py 273 line_number: {}'.format(i))
    return [future.result() for future in tqdm(futures) if future.result() is not None]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


# From https://github.com/r9y9/wavenet_vocoder/blob/master/utils.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def trim_silence(wav, hparams):
    '''Trim leading and trailing silence

	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
	'''
    # Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(wav, top_db=hparams.trim_top_db, frame_length=hparams.trim_fft_size,
                                hop_length=hparams.trim_hop_size)[0]


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")


def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    ''' short time Fourier transform (STFT)
    Transform audio file to complex-valued matrix
    input: time series form of audio file
    output: STFT matrix (numpy array)'''
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


def _istft(y, hparams):
    '''Inverse STFT
    Transform STFT matrix back to time series
    input STFT matrix
    output time series 1 dimensional numpy array'''
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
	"""
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
	"""
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectrogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * (
                        (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0,
                           hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * (
                    (S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (
                                 2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0,
                             hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (
                    2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)



def preprocess(args, input_folders, out_dir, hparams):
    '''Use datasets/preprocessor.py module to generate mel spectrogram and export to out_dir folder'''
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    linear_dir = os.path.join(out_dir, 'linear')
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(linear_dir, exist_ok=True)
    metadata = build_from_path(hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs,
                                            tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    '''write down metadata information to train.txt file in out_dir folder
        this metadata file holds:
            -training data audio file name,
            -mel spectrogram matrix file name,
            -linear spectrogram matrix file name,
            -length of audio time series array,
            -length of spectrogram matrix
            -and correspondent text.
        separated by |
        ex: 'speech-audio-00004.npy|speech-mel-00004.npy|speech-linear-00004.npy|114176|446|produced the block books, which were the immediate predecessors of the true printed book,'
    '''
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sr = hparams.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), mel_frames, timesteps, hours))
    print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
    print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
    print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


def norm_data(args):
    ''' checking whether inputed data is supported or not
    if valid data was passed, return input_folder path'''
    merge_books = (args.merge_books == 'True')
    print('Selecting data folders..')
    supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS']
    if args.dataset not in supported_datasets:
        raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
            args.dataset, supported_datasets))

    if args.dataset.startswith('LJSpeech'):
        return [os.path.join(args.base_dir, args.dataset)]

    if args.dataset == 'M-AILABS':
        supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
                               'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
        if args.language not in supported_languages:
            raise ValueError('Please enter a supported language to use from M-AILABS dataset! \n{}'.format(
                supported_languages))

        supported_voices = ['female', 'male', 'mix']
        if args.voice not in supported_voices:
            raise ValueError('Please enter a supported voice option to use from M-AILABS dataset! \n{}'.format(
                supported_voices))

        path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
        supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if args.reader not in supported_readers:
            raise ValueError('Please enter a valid reader for your language and voice settings! \n{}'.format(
                supported_readers))

        path = os.path.join(path, args.reader)
        supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        if merge_books:
            return [os.path.join(path, book) for book in supported_books]
        else:
            if args.book not in supported_books:
                raise ValueError('Please enter a valid book for your reader settings! \n{}'.format(
                    supported_books))
            return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)
    preprocess(args, input_folders, output_folder, hparams)


def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default= '') ### because runing from current folder, need to move out before loading data
    parser.add_argument('--hparams', default='',        ## modified hparams if neccessary
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', default='LJSpeech-1.1')        #audio data folder to be used
    parser.add_argument('--language', default='en_US')
    parser.add_argument('--voice', default='female')
    parser.add_argument('--reader', default='mary_ann')
    parser.add_argument('--merge_books', default='False')
    parser.add_argument('--book', default='northandsouth')
    parser.add_argument('--output', default='Tacotron_input')        #output folder - which holds Tacotron training data
    parser.add_argument('--n_jobs', type=int, default=cpu_count()) #number of cpu use for multiprocessing
    args = parser.parse_args()
    modified_hp = hparams.parse(args.hparams)
    assert args.merge_books in ('False', 'True')                    # add contrain to merge_book param
    run_preprocess(args, modified_hp)

if __name__ == '__main__':
    main()
