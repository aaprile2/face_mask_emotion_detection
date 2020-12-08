'''
LABELS

Creating ground truth csvs for the training and testing images.
'''

# Imports
import pandas as pd
import glob
import cv2

# Path of cropped dataset
dir = '/Users/aaprile/Desktop/cropped_archive'

# Create lists of filename and label, for training and testing
train_file = []
test_file = []
train_label = []
test_label = []
train_array = []
test_array = []

# Create training file
for label in glob.glob(dir + '/train/*'):
    for img in glob.glob(label + '/*'):
        train_file.append(img.split('/')[-1])
        train_label.append(label.split('/')[-1])
        train_array.append(list(cv2.imread(img,0).flatten()))
        

# Create testing file
for label in glob.glob(dir + '/test/*'):
    for img in glob.glob(label + '/*'):
        test_file.append(img.split('/')[-1])
        test_label.append(label.split('/')[-1])
        test_array.append(list(cv2.imread(img,0).flatten()))
   
# Create DataFrame
train = pd.DataFrame(list(zip(train_file, train_array, train_label)), 
               columns =['file_name', 'array', 'label'])

test = pd.DataFrame(list(zip(test_file, test_array, test_label)),
                    columns =['file_name', 'array', 'label'])

# Export to csv
train.to_csv(dir + '/cropped_train.csv', index=False)
test.to_csv(dir + '/cropped_test.csv', index=False) 
