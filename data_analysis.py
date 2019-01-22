

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random
import cv2
import os


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def img_generator(samples, batch_size=32, augment=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                ### randomnly sample left center or right image
                img_idx = np.random.randint(3)
                img_pth = batch_sample[img_idx]

                steering_angle = float(batch_sample[3])

                ### left image
                if img_idx == 0:
                    steering_angle += 0.2
                if img_idx == 2:
                    steering_angle -= 0.2
                
                img = cv2.imread(img_pth)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                ### data augmentation

                if augment:
                    img = augment_brightness_camera_images(img)
                    img, steering_angle = augment_flip_camera_images(img, steering_angle)
                
                img = img[50:-20,...]
                img = cv2.resize(img, (64, 64))
                
                images.append(img)
                angles.append(steering_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



def augment_brightness_camera_images(image):
    """
    https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
    """

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1

def augment_flip_camera_images(image, steering_angle):

    if np.random.uniform() > 0.5:
        
        image_flipped = np.fliplr(image)
        steering_angle_flipped = -steering_angle

        return (image_flipped, steering_angle_flipped)
    
    else:
        return (image, steering_angle)







csv_pth = './data/data_orig/driving_log.csv'

csv_headers = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

csv_data = pd.read_csv(csv_pth)
csv_data.columns = csv_headers

steering_data = csv_data['steering']

print(len(steering_data))



samples = []

data_dir = './data'
subdir = 'data_orig'

for row in csv_data.itertuples():

    left_img_nm = row.left.split('/')[-1]
    center_img_nm = row.center.split('/')[-1]
    right_img_nm = row.right.split('/')[-1]

    left_img_pth = os.path.join(data_dir, subdir, 'IMG', left_img_nm)
    center_img_pth = os.path.join(data_dir, subdir, 'IMG', center_img_nm)
    right_img_pth = os.path.join(data_dir, subdir, 'IMG', right_img_nm)



    samples.append((left_img_pth, center_img_pth, right_img_pth, row.steering))


sample_generator = img_generator(samples, batch_size=64)

augmented_angles = []

for i in range(len(samples)//64):
    print(i)
    _, angles = next(sample_generator)
    # print(angles.tolist())
    augmented_angles += angles.tolist()

# print(augmented_angles)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

ax1.hist(steering_data, bins=200, normed=True)
ax1.set_xlabel('bin of steering angle')
ax1.set_title('steering angle distribution of the original data')

ax2.hist(augmented_angles, bins=200, normed=True)
ax2.set_xlabel('bin of steering angle')
ax2.set_title('steering angle distribution of the augmented data')

plt.tight_layout()
plt.savefig('./data_analysis_hist.png')
plt.show()


