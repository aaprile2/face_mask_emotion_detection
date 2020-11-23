'''
LABELS

Creating ground truth csvs for the training and testing images.
'''

# Imports
import pandas as pd
import glob

# Path of cropped dataset
dir = '/Users/aaprile/Desktop/cropped_archive'

# Create lists of filename and label, for training and testing
train_file = []
test_file = []
train_label = []
test_label = []

# Create training file
for label in glob.glob(dir + '/train/*'):
    for img in glob.glob(label + '/*'):
        train_file.append(img.split('/')[-1])
        train_label.append(label.split('/')[-1])

# Create testing file
for label in glob.glob(dir + '/test/*'):
    for img in glob.glob(label + '/*'):
        test_file.append(img.split('/')[-1])
        test_label.append(label.split('/')[-1])
   
# Create DataFrame
train = pd.DataFrame(list(zip(train_file, train_label)), 
               columns =['file_name', 'label'])

test = pd.DataFrame(list(zip(test_file, test_label)),
                    columns =['file_name', 'label'])

# Export to csv
train.to_csv(dir + '/train.csv', index=False)
test.to_csv(dir + '/test.csv', index=False) 