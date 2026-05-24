from google.colab import drive
drive.mount('/content/drive')

%%capture
!pip install mne
!unzip "/content/drive/MyDrive/A Multi-Modal Dataset of EEG Signals and M-CHAT As.zip" -d .

%%capture
!pip install mne
!unzip "/content/drive/MyDrive/A Multi-Modal Dataset of EEG Signals and M-CHAT As.zip" -d .

import os
import glob
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

channels = [
    'Fz',
    'C3',
    'Cz',
    'C4',
    'Pz',
    'PO7',
    'Oz',
    'PO8'
]

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=250, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered = filtfilt(b, a, data, axis=0)

    return filtered

def create_windows(data, window_size=256, stride=128):

    windows = []

    for start in range(0, len(data) - window_size, stride):

        end = start + window_size

        segment = data[start:end]

        windows.append(segment)

    return np.array(windows)

# Unzipping the correct dataset that is available in /content/
!unzip "/content/ASD-EEG-MCHAT-Dataset (1).zip" -d /content/

# Now list the content of /content/ to find the newly unzipped folder
print("\nListing content of /content/ after unzipping:")
!ls -l /content/

# The paths have been identified from the unzipped folder name
asd_path = "/content/ASD-EEG-MCHAT-Dataset/Raw ASD Data/EEG Data"
ctrl_path = "/content/ASD-EEG-MCHAT-Dataset/Raw Control Data/EEG Data"

# The 'rm' commands are removed as the files do not exist at the specified path.
# !rm '/content/ASD-EEG-MCHAT-Dataset/Raw ASD Data/EEG Data/ASD_EEG_File_Info.csv'
# !rm '/content/ASD-EEG-MCHAT-Dataset/Raw Control Data/EEG Data/Control_EEG_File_Info.csv'

asd_files = sorted(
    glob.glob(os.path.join(asd_path, "*.csv"))
)

ctrl_files = sorted(
    glob.glob(os.path.join(ctrl_path, "*.csv"))
)

print("\nASD Files found:", len(asd_files))
print("CTRL Files found:", len(ctrl_files))

# Only attempt to print sample files if the lists are not empty
if asd_files:
    print("\nSample ASD File:")
    print(asd_files[0])
else:
    print("\nNo ASD files found.")

if ctrl_files:
    print("\nSample CTRL File:")
    print(ctrl_files[0])
else:
    print("\nNo CTRL files found.")

X = []
y = []

window_size = 256
stride = 128

# Filter to include only subject data files and exclude 'File_Info.csv'
# Also handle the 'ASD_Sunject_15.csv' typo.
filtered_asd_files = [
    file for file in asd_files
    if ('Subject' in os.path.basename(file) or 'Sunject' in os.path.basename(file)) and os.path.basename(file).endswith('.csv')
]
filtered_ctrl_files = [
    file for file in ctrl_files
    if 'Subject' in os.path.basename(file) and os.path.basename(file).endswith('.csv')
]

# Debugging prints to confirm filtered file lists
print("Debug: ASD Files found (before filtering):", len(asd_files))
print("Debug: CTRL Files found (before filtering):", len(ctrl_files))
print("Debug: Filtered ASD Files:", len(filtered_asd_files))
print("Debug: Filtered CTRL Files:", len(filtered_ctrl_files))
if filtered_asd_files: print("Debug: Sample Filtered ASD:", filtered_asd_files[0])
if filtered_ctrl_files: print("Debug: Sample Filtered CTRL:", filtered_ctrl_files[0])

for file in filtered_asd_files:

    print("Processing ASD file:", file)
    df = pd.read_csv(file)
    df.columns = channels
    eeg = df.values
    eeg = bandpass_filter(eeg)
    scaler = StandardScaler()
    eeg = scaler.fit_transform(eeg)
    windows = create_windows(
        eeg,
        window_size=window_size,
        stride=stride
    )
    X.extend(windows)
    y.extend([1] * len(windows))
for file in filtered_ctrl_files:
    print("Processing CTRL file:", file)
    df = pd.read_csv(file)
    df.columns = channels
    eeg = df.values
    eeg = bandpass_filter(eeg)
    scaler = StandardScaler()
    eeg = scaler.fit_transform(eeg)
    windows = create_windows(
        eeg,
        window_size=window_size,
        stride=stride
    )
    X.extend(windows)
    y.extend([0] * len(windows))

X = np.array(X)
y = np.array(y)

print("\nFinal Dataset Shape")
print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)
else:
    print("Error: X and y are empty. Please ensure data is loaded correctly in previous steps.")
    print("Specifically, check the unzipping and data loading steps as 'ASD Files found: 0' and 'CTRL Files found: 0' were reported in the notebook's output.")


model = Sequential()
model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=(256, 8)
    )
)

model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.3))
model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu'
    )
)

model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu'
    )
)

model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        y_pred
    )
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.grid(True)

plt.show()

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Train Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.grid(True)

plt.show()

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix\n")
print(cm)

auc = roc_auc_score(y_test, y_pred_proba)
print("\nROC-AUC Score:", auc)

