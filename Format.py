# Aprile, Amezquita, Schaber
# CPE-695 Final Project
# Facial Expression Detection with Limited Features

'''
DATA FORMATTING

Creating numpy array files of the training and testing datasets
for easier data transfer and import.
'''

# Imports
import pandas as pd
import glob
import numpy as np
import cv2
from keras.utils import to_categorical

# Path of cropped dataset
dir = '/Users/aaprile/Desktop/cropped_archive'

# Create lists of filename and label, for training and testing
train_label = []
test_label = []
train_array = []
test_array = []

# Create training file
for label in glob.glob(dir + '/train/*'):
    for img in glob.glob(label + '/*'):
        train_label.append(label.split('/')[-1])
        train_array.append(cv2.imread(img, 0))
        


# Create testing file
for label in glob.glob(dir + '/test/*'):
    for img in glob.glob(label + '/*'):
        test_label.append(label.split('/')[-1])
        test_array.append(cv2.imread(img, 0))
   
# Stack arrays
training_data = np.stack(train_array, axis=0)
testing_data = np.stack(test_array, axis=0)
training_labels = np.stack(train_label, axis=0)
testing_labels = np.stack(test_label, axis=0)

# Export arrays to files
np.save('/Users/aaprile/Desktop/cropped_train_data.npy', training_data)
np.save('/Users/aaprile/Desktop/cropped_test_data.npy', testing_data)
np.save('/Users/aaprile/Desktop/cropped_train_labels_1D.npy', training_labels)
np.save('/Users/aaprile/Desktop/cropped_test_labels_1D.npy', testing_labels)
np.save('/Users/aaprile/Desktop/cropped_train_labels_encoded.npy', to_categorical(training_labels))
np.save('/Users/aaprile/Desktop/cropped_test_labels_encoded.npy', to_categorical(testing_labels))