'''
IMAGE PROCESSING

Using dlib, finds facial landmarks for the nose in each images
and crops from that point up. Saves image to appropriate 
train/test split and class folders.

Requires download of FER-2013 dataset: https://www.kaggle.com/msambare/fer2013.
'''

# Imports
from imutils import face_utils
import numpy as np
import dlib
import cv2
import glob
import os

# Parent path of downloaded FER dataset
dir = '/Users/aaprile/Desktop/'

# Load predictor (using pretrained weights for dlib shape predictor)
predictor = dlib.shape_predictor("/Users/aaprile/Downloads/shape_predictor_68_face_landmarks.dat")


# Image cropping function
def crop_image(path, target_path):
    # Read image
    img = cv2.imread(path)
    
    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Define bounding box of face
    # Single faces are bounded, use top left and bottom right points
    face = dlib.rectangle(0,0, img.shape[0], img.shape[1])

    # Further define box points
    x1 = face.left()
    y1 = face.top()
    x2 = face.right() 
    y2 = face.bottom() 

    # Create landmark object
    landmarks = predictor(image=img, box=face)
    
    # Get y-coordinate of landmark point 31
    y = landmarks.part(30).y

    # Crop image and save to target path
    cv2.imwrite(target_path + '/cropped_' + path.split('/')[-1], img[0:29, :])
    
    

# Make folder for cropped images
# !!!! If rerunnning script, make sure to call shutil.rmtree(name)
#      or there will be an error
name = dir + 'cropped_archive/'

try:
    os.mkdir(name)
except:
    print('ERROR: Folder exists')
    

count = 0 

# Crop images, maintaining original directory structure
# (Includes progress messages)
for spt in glob.glob(dir + 'archive/*'):
    print('Cropping: ' + spt.split('/')[-1])
    
    # Create training and testing directory
    t_name = name + spt.split('/')[-1] + '/'
    os.mkdir(t_name)
    
    for label in glob.glob(spt + '/*'):
        print('\t Cropping: ' + label.split('/')[-1])
        
        # Create class directory
        c_name = t_name + label.split('/')[-1]
        os.mkdir(c_name)
        
        for pic in glob.glob(label + '/*'):
            try:
                crop_image(pic, c_name)
                count += 1
            except:
                print('ERROR: ' + pic + ' failed')


# Print cropping numbers for confirmation
print('Total images cropped: ', count)
            
