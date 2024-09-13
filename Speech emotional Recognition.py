import numpy as np
import pandas as pd

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

paths = []
labels = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

len(paths)

paths[:5]

labels[:5]



## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


df['label'].value_counts()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame (df) with a column 'label' containing categories
# If 'label' contains non-numeric data, use it directly for the plot
sns.countplot(data=df, x='label')

# If you want to plot the count of unique values, you don't need to convert it to float
# sns.countplot(data=df, x='label')

# Show the plot
plt.show()



def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectrogram(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max), y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Assuming you have a DataFrame (df) with columns 'speech' and 'label'
emotion = 'fear'
path = np.array(df['speech'][df['label'] == emotion])[0]  # This gets the first 'fear' audio file
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data, sampling_rate, emotion)

# Play the audio
Audio(path)




import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectrogram(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max), y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Assuming you have a DataFrame (df) with columns 'speech' and 'label'
emotion = 'angry'
path = np.array(df['speech'][df['label'] == emotion])[1]  # This gets the second 'angry' audio file
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectrogram(data, sampling_rate, emotion)

# Play the audio
Audio(path)



emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)



emotion = 'neutral'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)
emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)



emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)



emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)



def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


extract_mfcc(df['speech'][0])


X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))


X_mfcc
X = [x for x in X_mfcc]
X = np.array(X)
X.shape



## input split
X = np.expand_dims(X, -1)
X.shape


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])


y = y.toarray()


y.shape



from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(123, return_sequences=False, input_shape=(40,1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



pip install librosa soundfile



model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'E:\Speech model'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
# Save model and weights




history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)




epochs = list(range(100))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()



model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'E:/DATASET'
# Save model and weights
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import librosa
import librosa.display

# User uploads an audio file (assume it's named 'user_audio.wav')

# Preprocess user's audio
user_audio_path = '/kaggle/input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/OAF_disgust/OAF_bar_disgust.wav'  
# Path to the user's uploaded audio
user_data, user_sr = librosa.load(user_audio_path, duration=3, offset=0.5)  
# Adjust duration and offset as needed
user_mfcc = extract_mfcc(user_audio_path)  
# Extract MFCC features from user's audio

# Load the pre-trained model
# Replace 'your_model.h5' with the actual path to your pre-trained model
from keras.models import load_model
model = load_model('/kaggle/working/E:\Speech model/Emotion_Voice_Detection_Model.h5')

# Use the model to predict emotion
predicted_emotion = model.predict(np.expand_dims(user_mfcc, axis=0))

# Map the prediction to emotion label (e.g., 'happy', 'angry', etc.)
emotion_labels = ['happy', 'angry', 'disgust', 'fear', 'neutral', 'sad', 'ps']
predicted_emotion_label = emotion_labels[np.argmax(predicted_emotion)]

# Display the predicted emotion
print(f'Predicted Emotion: {predicted_emotion_label}')

# Display the waveform plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('Waveform', size=20)
librosa.display.waveshow(user_data, sr=user_sr)

# Create and display the spectrogram
plt.subplot(1, 2, 2)
plt.title('Spectrogram', size=20)
S = librosa.feature.melspectrogram(y=user_data, sr=user_sr)
S_db = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

plt.show()

