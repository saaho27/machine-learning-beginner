import tensorflow as tf 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow_io as tfio

model= load_model("/home/saaho/Mechine_learnig/DeepLearning/MODELS/Language_prediction/Language_model_gt/model_langv_1_5.h5")

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def preprocess_test(file_path): 
    wav =load_wav_16k_mono(file_path)
   
    wav = wav[:64000]
    zero_padding = tf.zeros([64000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
 
    spectrogram = tf.signal.stft(wav, frame_length=512, frame_step=256)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def prediction():
    audio_file = input("Enter The Language Audio file to check Which Language:")
    preprocess_audio = preprocess_test(audio_file)




    audio_to_predict = np.reshape(preprocess_audio , (-1, 249, 257, 1))
    prediction  = model.predict(audio_to_predict)
    if np.any(prediction  > 0.50):

        ypred_labels = np.argmax(prediction , axis=1)

    
        mapping = {0: 'Telugu', 1: 'Tamil', 2: 'Malayalam', 3: 'Hindi'}
        predicted_languages = [mapping[label] for label in ypred_labels]

        print(f"The model predicted this language is {predicted_languages[0]} Language")
    else:
        print("Language not found")
if __name__ == "__main__":
    while True:
       prediction()
