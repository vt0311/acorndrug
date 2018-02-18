import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob

################
### Overview ###
################
base_bone_dir = 'C:/BoneAge/'
print(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))

age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
age_df['path'] = age_df['id'].map(
    lambda x: base_bone_dir + 'boneage-training-dataset/{}.png'.format(x) ) 

age_df['exists'] = age_df['path'].map(os.path.exists)
print(age_df) 
#Return True if path refers to an existing path or an open file descriptor.

print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')

boneage_mean = age_df['boneage'].mean()
boneage_div = 2*age_df['boneage'].std()
# we don't want normalization for now
#boneage_mean = 0
#boneage_div = 1.0
age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
age_df.dropna(inplace = True)
age_df.sample(3)

#age_df[['boneage', 'male', 'boneage_zscore']].hist(figsize = (10, 5))

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
#train_df[['boneage', 'male']].hist(figsize = (10, 5))
#plt.show()

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

#IMG_SIZE = (224, 224) # default size for inception_v3
IMG_SIZE = (299, 299)
core_idg = ImageDataGenerator(
    samplewise_center=False, # Set each sample mean to 0
    samplewise_std_normalization=False, # Divide each input by its std
    horizontal_flip = True, # Randomly flip inputs horizontally
    vertical_flip = False, # Randomly flip inputs vertically
    height_shift_range = 0.15, # Range for random horizontal shifts
    width_shift_range = 0.15, # Range for random vertical shifts
    rotation_range = 5, # Degree range for random rotations
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
                # batch_size란 : 뭔가를 하나 수정할 때 모든 데이터를 다시 처리하는 것은 낭비이므로, 
                #        훈련 데이터를 여러 개의 작은 배치로 나누어 매개변수를 수정하는데, 이떄의 작은 배치 사이즈를 말한다.
valid_gen = flow_from_dataframe(
                core_idg, valid_df, path_col = 'path', y_col = 'boneage_zscore', 
                target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 256) 
                # we can use much larger batches for evaluation
                
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next( flow_from_dataframe(
                        core_idg, valid_df, path_col = 'path', y_col = 'boneage_zscore', 
                        target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 1024)) 
                        # one big batch

t_x, t_y = next(train_gen)
#fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
#for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
#    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
#    c_ax.set_title('%2.0f months' % (c_y*boneage_div+boneage_mean))
#    c_ax.axis('off')
#plt.show()    

# print(t_x)
print(t_x.shape)
# print(t_y)
print(t_y.shape)
    
###########################
###Create a Simple Model###
###########################
import keras
from keras.applications.inception_v3 import InceptionV3
#from inception_v4 import inception_v4
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from IPython.display import clear_output

base_iv3_model = InceptionV3(input_shape =  t_x.shape[1:],
                              include_top = False, 
                              weights = 'imagenet')
'''
base_iv3_model = inception_v4(num_classes =  1000,
                              dropout_keep_prob = 0,
                              weights = 'imagenet',
                              include_top = False)
'''
# include_top : whether to include the fully-connected layer at the top of the network.
# weights : one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
base_iv3_model.trainable = False


bone_age_model = Sequential()

bone_age_model.add(base_iv3_model)
bone_age_model.add(GlobalAveragePooling2D())
bone_age_model.add(Dropout(0.5))
bone_age_model.add(Dense(1024, activation = 'tanh'))
bone_age_model.add(Dropout(0.25))
bone_age_model.add(Dense(1, activation = 'linear')) # linear is what 16bit did
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

from keras.metrics import mean_absolute_error
def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)

bone_age_model.compile(optimizer = 'adam', loss = 'mse', metrics = [mae_months])
# metrics :  a function that is used to judge the performance of your model. 
# bone_age_model.summary()

################
###Model Save###
################
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('bone_age_v3')

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
                      patience=5) # probably needs to be more patient, but kaggle time is limited

# updatable plot
# a minimal example (sort of)
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

callbacks_list = [checkpoint, early, reduceLROnPlat, plot_losses]

####################
###Model Training###
####################
bone_age_model.fit_generator(
    train_gen, 
    steps_per_epoch=10, # Total number of steps (batches of samples) to yield from generator
                         # 생성기에서 얻는 총 단계 수 (샘플 배치)입니다.
                         # It should typically be equal to the number of samples 
                         # of your dataset divided by the batch size
                         # 일반적으로 데이터 셋의 샘플 수를 배치 사이즈로 나눈 값과 같아야 합니다. ( = 데이터셋 샘플수 / 배치사이즈 )
    validation_data = (test_X, test_Y), # This can be either
                                        # A generator for the validation data
                                        # A tuple (inputs, targets)
                                        # A tuple (inputs, targets, sample_weights)
    epochs = 2, #  total number of iterations on the data # 데이터의 총 반복 횟수
    callbacks = callbacks_list # List of callbacks to be called during training.
    #callbacks=[plot_losses]
    ) 

##########################
###Evaluate the Results###
##########################
bone_age_model.load_weights(weight_path)
pred_Y = boneage_div*bone_age_model.predict(test_X, batch_size = 32, verbose = True)+boneage_mean
test_Y_months = boneage_div*test_Y+boneage_mean
                        
# Finally, we trained the final model with a minibatch size of 16 for 500 epochs (approximately 50 hours) with the ADAM optimizer attempting to minimize mean absolute error of the output. We reduced the learning rate when the validation loss plateaued.                       
# 마지막으로 ADAM 최적화 프로그램이 출력의 평균 절대 오류를 최소화하려고 시도하면서 500 에포크 (약 50 시간) 동안 미니 배치 크기 16으로 최종 모델을 교육했습니다. 검증 손실이 극대화되면 학습률이 감소합니다.
'''
# summarize history for mae_month
plt.plot(history.history['mae_month'])
plt.plot(history.history['val_mae_month'])
plt.title('model mae_month')
plt.ylabel('mae_month')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''






                        