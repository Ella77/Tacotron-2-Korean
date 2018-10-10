# Tacotron-2-Korean
The implementation of Tacotron 2 on Korean language dataset (KSS, Zeroth_Korean..)

 Datasets:
 	
	https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
	http://www.openslr.org/40/
	
remember to reformat the transcript file to match the format of this repo transcript.txt. Or you can change the code to match your file format.

Preparing data and transcript as the format of samples in LJSpeech-1.1 folder

Step 1: Preprocessing audio.
  
		python Utils/AudioProcessing/AudioPreprocess.py

Above command will process wav files and transcript.txt file in LJSpeech-1.1 folder and transform them to numpy array, then store in Tacotron_input folder. 

Step 2: Train Tacotron

		python TacotronModel/train.py
		 
This command trains the Tacotron model using data from Tacotron_input folder. The model will be saved  by interval in TacotronModel/train.py arguments. 
If you do not want to resume the training, use: 

		python TacotronModel/train.py --restore=False
		
		
Step 3: Synthesize Tacotron

		python TacotronModel/Synthesize.py
		
Using Tacotron mode, this command synthesizes data (in Tacotron_input folder) to mel spectrograms (will be stored in tacotron_output/gta folder). The tacotron_output folder will hold input data of wavenet_vocoder

Step 4: Train Wavenet

		python wavenet_vocoder/train.py
	
This command using data in tacotron_output folder as input, to train a wavenet model to synthesize the audio. When training, you will see the synthesized wav of training (in wavenet_trained_logs/wavs) has much more quality compare to synthesized wav of evaluating (in wavenet_trained_logs/eval/wavs). Don't worry, because the training uses GTA but evaluating does not. 

Step 5: inference
	
	5-1: Tacotron inference:
		
		python TacotronModel/Synthesize.py --mode=inference
	
This command will get the text (sentences) in Hyperparam.py file and predict the mel spectrogram, then store in tacotron_output/inference folder
		
	5-2: Wavenet inference
	
		python wavenet_vocoder/synthesize.py

This command get the data in tacotron_output/inference folder, using wavenet pretrained model (in wavenet_trained_logs/wavenet_pretrained) to synthesize audio, this step could take a while (30 minutes or more). I'm finding the way to apply Paralell Wavenet to improve this.
The synthesized audios will be stored in wavenet_output folder.

Note:
If your GPU has less memory:
1. reduce the clip_mels_length parameter in Hyperparam.py (this will skip long files) or split audio into smaller parts before training. 
2. increase outputs_per_step (recommended maximum is 3)
3. reduce mel_channels <-- this does reduce the predicted mel spectrogram quality. 
4. ask your wife for money and buy a good one.

I wanted to upload pretrained checkpoint files so you can save training time. But Guthub does not allow uploadding large file as checkpoin. So, you can delete 'Tacotron_pretrained_logs' and 'wavenet_pretrained_logs' and train from scratch. Or contact me to get the pretrained checkpoint files.

Becareful of creating sentences.txt file with advanced editors, It might generate a strange format character at the begining. Recommend to use plain notepad

output examples here: https://clyp.it/user/nspiu1ef

Reference: 

<list>
https://github.com/Rayhane-mamah

https://deepmind.com/blog/wavenet-generative-model-raw-audio/

https://github.com/r9y9/wavenet_vocoder

https://github.com/keithito/tacotron
</list>
