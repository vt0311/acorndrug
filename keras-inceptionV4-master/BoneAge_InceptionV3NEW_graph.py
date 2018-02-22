'''
Created on 2018. 2. 21.

@author: acorn
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
# from skimage.io import imread
import os
from glob import glob

################
### Overview ###
################
base_bone_dir = 'D:/BoneAge/'
print(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))

age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
age_df['path'] = age_df['id'].map(
    lambda x: base_bone_dir + 'boneage-training-dataset/{}.png'.format(x) ) 

age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df) 
#Return True if path refers to an existing path or an open file descriptor.

print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')

boneage_mean = age_df['boneage'].mean() #정규화 (평균 표준편차)
boneage_div = 2*age_df['boneage'].std()
# we don't want normalization for now
boneage_mean = 0 #정규화안하고 함 
boneage_div = 1.0
age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
age_df.dropna(inplace = True)
age_df.sample(3)

# age_df[['boneage', 'male', 'boneage_zscore']].hist(figsize = (10, 5))

age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
# print(age_df)


###############################################
### Split Data into Training and Validation ###
###############################################
from sklearn.model_selection import train_test_split
raw_train_df, valid_df = train_test_split(age_df, test_size = 0.25, 
                                          random_state = 2018, 
                                          stratify = age_df['boneage_category'])
# random_state : the seed used by the random number generator
# stratify (계층화하다) : data is split in a stratified fashion, using this as the class labels.

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
#age_df[['boneage', 'male']].hist(figsize = (10, 5))

### Balance the distribution in the training set ###
train_df = raw_train_df.groupby(['boneage_category', 'male'])
train_df = train_df.apply(lambda x: x.sample(500, replace = True))
# sample : Sample with or without replacement. (default = False)
train_df = train_df.reset_index(drop = True)
# print(train_df)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
# train_df[['boneage', 'male']].hist(figsize = (10, 5))
# plt.show()

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

IMG_SIZE = (224, 224) # default size for inception_v3
#IMG_SIZE = (256, 256)
core_idg = ImageDataGenerator(
    samplewise_center=False, # Set each sample mean to 0
    samplewise_std_normalization=False, # Divide each input by its std
    horizontal_flip = True, # Randomly flip inputs horizontally
    vertical_flip = False, # Randomly flip inputs vertically
    height_shift_range = 0.2, # Range for random horizontal shifts
    width_shift_range = 0.2, # Range for random vertical shifts
    rotation_range = 20, # Degree range for random rotations
    shear_range = 0.01, # Shear(깍다) angle in counter-clockwise direction as radians 
    fill_mode = 'reflect',
    # One of {"constant", "nearest", "reflect" or "wrap"}. 
    # Points outside the boundaries of the input are filled according to the given mode:
    zoom_range=0.25, # Range for random zoom
    preprocessing_function = preprocess_input) # function that will be implied on each input. 

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse',**dflow_args)
    # flow_from_directory
    # -> Takes the path to a directory, and generates batches of augmented/normalized data.
    # sparse : 드문, 희박한 ( "sparse" will be 1D integer labels)
    # **dflow_args
    #    -> target_size : The dimensions to which all images found will be resized.
    #    -> color_mode : one of "grayscale", "rbg"
    #    -> batch_size : size of the batches of data (default: 32)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values) # Join a sequence of arrays along a new axis
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_gen = flow_from_dataframe(
                core_idg, train_df, path_col = 'path', y_col = 'boneage_zscore', 
                target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 8)
                # (target_size, color_mode, batch_size)
#print('train_gen.history.keys:', train_gen.history.keys())
valid_gen = flow_from_dataframe(
                core_idg, valid_df, path_col = 'path', y_col = 'boneage_zscore', 
                target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 256) 
                # we can use much larger batches for evaluation
                
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next( flow_from_dataframe(
                        core_idg, valid_df, path_col = 'path', y_col = 'boneage_zscore', 
                        #target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 1024))
                         target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 128))  
                        # one big batch

t_x, t_y = next(train_gen)
#fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
#for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
#    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
#    c_ax.set_title('%2.0f months' % (c_y*boneage_div+boneage_mean))
#    c_ax.axis('off')

# print(t_x)
print(t_x.shape)
# print(t_y)
print(t_y.shape)
    
###########################
###Create a Simple Model###
###########################
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input
from keras.optimizers import adam, SGD
from keras.models import Sequential, Model

base_model =  InceptionV3(weights='imagenet', include_top=False)
input = Input(shape=(*IMG_SIZE, 3))
output_invV3 = base_model(input)
x = Flatten()(output_invV3)
x = Dense(512, activation='relu')(x)
predictions = Dense(1)(x)
model = Model(inputs=input, outputs=predictions)

from keras.metrics import mean_absolute_error
def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)
myOptimizer = adam(lr = 1e-3)
model.compile(optimizer = myOptimizer, loss = 'mse', metrics = [mae_months])
'''
Layer (type)                 Output Shape              Param #   
=================================================================
inception_v3 (Model)         (None, 5, 5, 2048)        21802784 
Param is obtained as : input values * neurons in the layer + bias values 
_________________________________________________________________
global_average_pooling2d_1 ( (None, 2048)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              2098176   
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 1025      
=================================================================
Total params: 23,901,985
Trainable params: 2,099,201
Non-trainable params: 21,802,784
'''

#from keras import optimizers

# sgd = optimizers.SGD(lr=0.005)
# bone_age_model.compile(optimizer = sgd, loss = 'mse', metrics = [mae_months])
# metrics :  a function that is used to judge the performance of your model. 
# bone_age_model.summary()

################
###Model Save###
################
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(
                weight_path, # filepath : string, path to save the model file
                monitor='val_loss', # monitor : quantity to monitor
                verbose=1, # verbosity mode, 0 or 1
                save_best_only=True, # if save_best_only=True -> the latest best model \ 
                                     # according to the quantity monitored will not be overwritten.
                mode='min', # one of {auto, min, max}, the decision to overwrite 
                            # the current save file is made based on the minimization 
                            # of the monitored quantity.     
                save_weights_only = True) #  only the model's weights will be saved

reduceLROnPlat = ReduceLROnPlateau(
                    monitor='val_loss', # quantity to be monitored.
                    factor=0.8, # factor by which the learning rate will be reduced
                    patience=10, # number of epochs with no improvement 
                                 # after which learning rate will be reduced.
                    verbose=1, # 0: quiet, 1: update messages
                    mode='auto', # one of {auto, min, max} In min mode, lr will be reduced 
                                 # when the quantity monitored has stopped decreasing
                    epsilon=0.0001, # threshold(문턱) for measuring the new optimum(최적의결과), 
                                    # to only focus on significant changes. 
                    cooldown=5, # number of epochs to wait before resuming normal operation 
                                # after lr has been reduced.
                    min_lr=0.0001) # lower bound on the learning rate.

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=50) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

####################
###Model Training###
####################
model.fit_generator(
    train_gen, 
    steps_per_epoch=1, # Total number of steps (batches of samples) to yield from generator
                         # It should typically be equal to the number of samples 
                         # of your dataset divided by the batch size
    validation_data = (test_X, test_Y), # This can be either
                                        # A generator for the validation data
                                        # A tuple (inputs, targets)
                                        # A tuple (inputs, targets, sample_weights)
    epochs = 1, #  total number of iterations on the data
    callbacks = callbacks_list ) # List of callbacks to be called during training.

history = model.fit(test_X, test_Y, validation_split=0.33, epochs=10, batch_size=10, verbose=0)
print('history.history.keys:', history.history.keys())
#history.history.keys: dict_keys(['val_loss', 'val_mae_months', 'loss', 'mae_months', 'lr'])

# summarize history for accuracy
plt.plot(history.history['mae_months'])
plt.plot(history.history['val_mae_months'])
plt.title('model mae_months')
plt.ylabel('mae_months')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

##########################
###Evaluate the Results###
##########################
model.load_weights(weight_path)
pred_Y = boneage_div*model.predict(test_X, batch_size = 32, verbose = True)+boneage_mean
test_Y_months = boneage_div*test_Y+boneage_mean
                        