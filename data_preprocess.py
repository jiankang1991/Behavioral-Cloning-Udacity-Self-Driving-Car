

import os
import cv2
import pandas as pd
import numpy as np
import random
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def augment_brightness_camera_images(image):
    """
    https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
    """

    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2BGR)

    return image1


def augment_trans_camera_images(image, trans_range):
    """
    https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc
    """
    rows,cols,ch = image.shape

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image

def augment_flip_camera_images(image, steering_angle):

    if np.random.uniform() > 0.5:
        
        image_flipped = np.fliplr(image)
        steering_angle_flipped = -steering_angle

        return (image_flipped, steering_angle_flipped)
    
    else:
        return (image, steering_angle)


if __name__ == "__main__":
    
    csv_pths = ['./data/data_orig/driving_log.csv']

    ### in case does not have
    csv_headers = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    example_num = 8

    plt.figure(figsize = (15,10))
    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(wspace=0.2, hspace=0.1) # set the spacing between axes. 

    img_num = 0

    for csv_pth in csv_pths:

        csv_data = pd.read_csv(csv_pth)
        csv_data.columns = csv_headers

        subdir = csv_pth.split('/')[-2]
        
        # print(csv_data)
        # print(csv_data['center'])
        # print(csv_data.columns)

        for idx, center_img_pth in enumerate(csv_data['center'][:example_num]) :

            # print(center_img_pth)
            img_pth = os.path.join('./data', subdir, 'IMG', center_img_pth.split('/')[-1])
            # print(img_pth)
            img = cv2.imread(img_pth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # print(img)

            img_nm = center_img_pth.split('/')[-1]
            title = img_nm.split('.')[0] + ' \n steering: {:.2f}'.format(csv_data['steering'][idx])


            ### data augmentation

            img = augment_brightness_camera_images(img)
            img, _ = augment_flip_camera_images(img, 0)

            ### crop

            img = img[50:-20,...]
            img = cv2.resize(img, (64, 64))

            # img = augment_trans_camera_images(img, 10)

            ax = plt.subplot(gs1[img_num])
            ax.set_axis_off()
            ax.imshow(img)
            # ax.imshow(img)
            ax.set_title(title)

            img_num += 1

    # plt.title('img_preprocessing_images')
    # plt.tight_layout()
    plt.savefig('./img_preprocessing_examples.png')
    plt.show()



















