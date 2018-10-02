import tensorflow as tf
import numpy as np
# Default hyperparameters
hparams = tf.contrib.training.HParams(
    #Loss and optimizing params
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer beta3 parameter
    mask_decoder = False,
    leaky_alpha=0.4,


    #### teacher forcing parameters
    tacotron_decay_learning_rate=True,
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=40000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.2,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate
    tacotron_teacher_forcing_mode = 'scheduled',    # mode of teacher forcing Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1., # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    tacotron_teacher_forcing_init_ratio=1.,  # initial teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_final_ratio=0.,  # final teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_start_decay=10000,# starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_steps=280000,# Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_alpha=0.,  # teacher forcing ratio decay rate. Relevant if mode='scheduled'
    tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=True,

##### machine learning network params
    conv_num_layers = 3,
    conv_kernel_size = (5,), # convolution filter size
    conv_channels = 512,        #convolution channels
    enc_lstm_hidden_size = 256, #lstm hiddent units size
    enc_conv_layers = 5,         #number of layers of convolutional network
    enc_conv_droprate=0.5,      #convolutional network droprate
    zoneout_rate = 0.1,         #zoneout rate for zoneoutLSTM network
    dropout_rate = 0.5,
    attention_dim =128,
    attention_filters = 32,
    attention_kernel=(31,),
    embedding_dim=512,          # dimensions of embedded vector
    postnet_kernel_size=(5,),   #postnet convolutional layers filter size
    postnet_channels = 512,     #postnet convolution channels of each layer
    postnet_num_layers = 5,     #number of postnet convolutional layers


    #text preprocessing params
    hangul_type = 1,        #type of hangul conversion
    ##audio preprocessing params
    trim_fft_size = 512,    ## use to trim silent part of M-AILABS dataset
    trim_hop_size = 128,    # use to trim silent part of M-AILABS dataset
    trim_top_db = 60,   # use to trim silent part of M-AILABS dataset
    rescale=True,  # whether rescale audio data before processing
    rescaling_max = 0.999,    #max scaling (if rescale is true, this parameter will be used)
    trim_silence = True,        #whether trim out silent parts
    sample_rate=44100, #22050, 44100
    fmin=25,  # min of voice frequency
    fmax=7600,  # max of voice frequency
    min_level_db=-100,
    ref_level_db=20,
    silence_threshold = 2,  #threshold of tobe trimmed silence audio

    #linear spectrogram params
    power=1.2,
    griffin_lim_iters=60,
    ### mel spectrogram params
    predict_linear = False,         #whether model predicts linear spectrogram
    signal_normalization=True,
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
    max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
    num_mels=100,  # Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq=513,  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    clip_mels_length=True,  # For cases of OOM (Not really recommended, working on a workaround)
    max_mel_frames=1000,  # Only relevant when clip_mels_length = True
    n_fft=1024,  # Extra window size is filled with 0 padding to match this parameter
    hop_size=256,  # For 22050Hz, 275 ~= 12.5 ms
    win_size=1024,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) keep win_size/sample_rate ratio at 20~40 ms
    frame_shift_ms=None,
    # wavenet paramters0
    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    input_type="mulaw-quantize",
    #if input_type = mulaw_quantize then:
    quantize_channels=256,
    #lws: local weighted sum
    use_lws=True,
    cin_channels= 100, ### requires cin_channel = num_mels
    #number of dilated convolutional network layers
    layers = 30, #######################################################################################################################
    # group convolutional layers to [number] of stack
    stacks = 3,   #######################################################################################################################
    out_channels= 256,
    residual_channels=512,
    gate_channels = 512,
    kernel_size = 3,
    skip_out_channels = 256,
    wavenet_dropout = 0.05,
    use_bias = True,
    gin_channels = -1,
    max_time_sec=None,
    max_time_steps=10000,  # Max time steps in audio used to train wavenet (decrease to save memory)
    upsample_conditional_features=True,# Whether to repeat conditional features or upsample them (The latter is recommended)
    upsample_scales=[16, 16],  # prod(scales) should be equal to hop size
    freq_axis_kernel_size=3,
    log_scale_min=float(np.log(1e-14)),
    normalize_for_wavenet=True,
#training, evaluating, synthesizing params
    #Tacotron training params
    tacotron_synthesis_batch_size = 32 * 16,
    tacotron_data_random_state = 1324,
    outputs_per_step = 2,
    tacotron_random_seed=45454,
    tacotron_swap_with_cpu = False,
    tacotron_test_size=0.1,
    tacotron_test_batches=None,
    tacotron_batch_size=48,
    max_iters = 3000,
    stop_at_any=True,
    # wavenet Training params
    wavenet_synthesis_batch_size = 3 * 2,
    wavenet_random_seed=5339,  # S=5, E=3, D=9 :)
    wavenet_swap_with_cpu=False, # Whether to use cpu as support to gpu for decoder computation
    wavenet_batch_size=2,  # batch size used to train wavenet.
    wavenet_test_size=0.0441,  # % of data to keep as test data, if None, wavenet_test_batches must be not None
    wavenet_test_batches=None,  # number of test batches.
    wavenet_data_random_state=1234,  # random state for train test split repeatability
    wavenet_learning_rate=1e-3,
    wavenet_adam_beta1=0.9,
    wavenet_adam_beta2=0.999,
    wavenet_adam_epsilon=1e-6,
    wavenet_ema_decay=0.9999,  # decay rate of exponential moving average
    train_with_GTA=False,  # Whether to use GTA mels to train wavenet instead of ground truth mels.
    ## environment params
    base_dir='',
    sentences=[
        '옛날 어느 마을에 한 총각이 살았습니다.',
        '사람들이 그 총각을 밥벌레 장군이라고 놀렸습니다.',
        '왜냐하면, 매일 밥만 먹고 빈둥빈둥 놀기 때문입니다.',
        '밥도 한두 그릇 먹는 게 아니라 ',
        '가마솥 통째로 먹어 치웁니다.',
        '그래서 몸집도 엄청 큽니다.',
        '그런데 힘은 어찌나 없는지 밥그릇 하나도 들지 못합니다.',
        '밥만 먹고 똥만 싸니 집안 살림이 거덜 나게 생겼습니다.',
        '걱정된 부모님은 밥벌레 장군을 불러 놓고 말했습니다.',
        '"얘야, 더는 널 먹여 살릴 수가 없구나."',
        '"집을 나가서 네 힘으로 살아 보아라"',
        '집을 나온 밥벌레 장군은 밥을 얻어먹고 다녔습니다.',
        '하루는 깊은 산골의 한 초가집을 지나가고 있었습니다.',
        '"여기서 밥을 얻어먹을 수 있으면 좋겠다."',
        '담 너머를 기웃거리고 있는데 아낙네가 나오더니 말했습니다.',
        '"이리 들어오시지요!"',
        '아낙네는 남편이 호랑이한테 잡아 먹혀 세 아들이',
        '호랑이를 잡으러 날마다 산에 간다는 이야기를 해주었습니다.    ',
    ]

    ### test text
# Eval sentences (if no eval file was specified, these sentences are used for eval)
)
