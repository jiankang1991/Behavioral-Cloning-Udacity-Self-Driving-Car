import keras
import sklearn
from sklearn.model_selection import train_test_split
import random
import os
import cv2
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Activation, Lambda, MaxPooling2D, Cropping2D, GlobalAveragePooling2D
from keras import optimizers
from keras.backend import tf as ktf
from keras.callbacks import ModelCheckpoint, TensorBoard


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
#                     if abs(steering_angle) > 1:
#                         steering_angle += 0.25
#                     else:
#                         steering_angle += 0.2
                    
                    steering_angle += 0.2
                    
                if img_idx == 2:
#                     if abs(steering_angle) > 1:
#                         steering_angle -= 0.25
#                     else:
#                         steering_angle -= 0.2
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


def read_csvs(csv_pths, data_dir):

    csv_headers = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    samples = []

    for csv_pth in csv_pths:

        csv_data = pd.read_csv(csv_pth)
        csv_data.columns = csv_headers

        subdir = csv_pth.split('/')[-2]

        #### namedtuple
        for row in csv_data.itertuples():
            
            left_img_nm = row.left.split('/')[-1]
            center_img_nm = row.center.split('/')[-1]
            right_img_nm = row.right.split('/')[-1]

            left_img_pth = os.path.join(data_dir, subdir, 'IMG', left_img_nm)
            center_img_pth = os.path.join(data_dir, subdir, 'IMG', center_img_nm)
            right_img_pth = os.path.join(data_dir, subdir, 'IMG', right_img_nm)

            samples.append((left_img_pth, center_img_pth, right_img_pth, row.steering))

    return samples



wd = 1e-4
dr_rate = 1
batch_sz = 256
# lr_rate=0.001
lr_rate = 0.001
epoch_num = 20

csv_pths = ['/opt/carnd_p3/data/driving_log.csv']

# csv_pths = ['/opt/carnd_p3/data_track2/driving_log.csv']


samples = read_csvs(csv_pths, '/opt/carnd_p3')

# print(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = img_generator(train_samples, batch_sz)
val_generator = img_generator(validation_samples, batch_sz)


inputs = Input(shape=(64, 64, 3))

x = Lambda((lambda x: x/255.0))(inputs)

# x = Cropping2D(cropping=((50,20), (0,0)))(x)

# x = Lambda(lambda x: ktf.image.resize_images(x, (64, 64)))(x)

x = Conv2D(3, [1,1], activation='relu', kernel_regularizer=keras.regularizers.l2(wd))(x)

x = Conv2D(16, [3,3], activation='relu', dilation_rate=dr_rate,  kernel_regularizer=keras.regularizers.l2(wd))(x)
x = MaxPooling2D()(x)
# x = Dropout(0.5)(x)

# x_1 = Flatten()(x)
# x_1 = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(wd))(x_1)

x = Conv2D(32, [3,3], activation='relu', dilation_rate=dr_rate,  kernel_regularizer=keras.regularizers.l2(wd))(x)
x = MaxPooling2D()(x)
# x = Dropout(0.5)(x)

x = Conv2D(64, [3,3], activation='relu', dilation_rate=dr_rate,  kernel_regularizer=keras.regularizers.l2(wd))(x)
x = MaxPooling2D()(x)
# x = Dropout(0.5)(x)

x = Conv2D(64, [3,3], activation='relu', dilation_rate=dr_rate, padding='SAME', kernel_regularizer=keras.regularizers.l2(wd))(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dropout(0.5)(x)

# x = Conv2D(512, [1,1], activation='relu', kernel_regularizer=keras.regularizers.l2(wd))(x)
# x = GlobalAveragePooling2D()(x)


x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(1, activation='linear')(x)

# x = Flatten()(x)
# x = Dense(1024)(x)

model = Model(inputs=inputs, outputs=x)
model.summary()


opt = optimizers.adam(lr=lr_rate)

model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])

# check_callback = ModelCheckpoint('./output/my_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

check_callback = ModelCheckpoint('./output_track2/my_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# vis_callback = TensorBoard(log_dir='./output/logs', histogram_freq=0, write_graph=True, write_images=True)
vis_callback = TensorBoard(log_dir='./output_track2/logs', histogram_freq=0, write_graph=True, write_images=True)

train_hist = model.fit_generator(train_generator, callbacks=[check_callback, vis_callback], epochs=epoch_num, steps_per_epoch=len(train_samples)//batch_sz,
                                 validation_data=val_generator, validation_steps=len(validation_samples)//batch_sz, verbose=1)


# model.save('./output/my_model.h5')








