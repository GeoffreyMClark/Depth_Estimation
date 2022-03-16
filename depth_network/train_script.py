import pandas as pd
import dataset_prep
import depth_prediction_net
import loss
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import EfficientNetB0

TF_FORCE_GPU_ALLOW_GROWTH=True


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

############################# settings for the training routine ####################################
width = 160
height = 90
batch_size = 12

get_dataset = dataset_prep.get_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


############################# Check the dataloader ####################################

train_data, depth_data = get_dataset.select_batch(batch_size)

train_gen = get_dataset.train_generator(batch_size)
validation_gen = get_dataset.validation_generator(batch_size)



############################# build the model ####################################

model_1
opt = Adam(lr=1e-5)
Disp_ResNet_autoencoder = get_depth_net.res_50_disp_autoencoder(height, width, 3)
Disp_ResNet_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
print(Disp_ResNet_autoencoder.summary())


# model_2
opt = Adam(lr=1e-5)
model_2 = get_depth_net.ResNet_block_autoencoder(height, width, 3)
model_2.compile(optimizer=opt, loss=get_loss.autoencoder_loss)
# print(model_2.summary())


# model_3
opt = Adam(lr=1e-5)
model_3 = get_depth_net.ResNet_resblock_disp_autoencoder(height, width, 3)
model_3.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
# print(model_3.summary())


# model_4
opt = Adam(lr=1e-5)
model_4 = get_depth_net.ours_autoencoder(height, width, 3)
model_4.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
# print(model_4.summary())


# model_5
opt = Adam(lr=1e-5)
model_5 = get_depth_net.res_50_disp_autoencoder(height, width, 3)
model_5.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
# print(Disp_ResNet_autoencoder.summary())


# model_6
opt = Adam(lr=1e-5)
model_6 = get_depth_net.ResNet_block_autoencoder(height, width, 3)
model_6.compile(optimizer=opt, loss=get_loss.autoencoder_loss)
# print(Disp_ResNet_autoencoder.summary())


# model_7
opt = Adam(lr=1e-5)
model_7 = get_depth_net.ResNet_resblock_disp_autoencoder(height, width, 3)
model_7.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
# print(model_7.summary())



# model_8
opt = Adam(lr=1e-5)
model_8 = get_depth_net.ours_autoencoder(height, width, 3)
model_8.compile(optimizer=opt, loss=get_loss.autoencoder_loss, loss_weights= [1/8, 1/4, 1/2])
# print(model_8.summary())



# Efficientnet model
opt = Adam(lr=1e-5)
model_2 = get_depth_net.Efficient_autoencoder(height, width, 3)
# model_2.compile(optimizer=opt, loss=get_loss.autoencoder_loss)
# print(model_2.summary())


############################# start training ####################################

# # start training -- model_v1
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v1/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# Disp_ResNet_autoencoder.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v2
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v2/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_2.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v3
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v3/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_3.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v4
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v4/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_4.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v5
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v5/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_5.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v6
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v6/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_6.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])


# # start training -- model_v7
# mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v7/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# # NAME = "depth_net_2.0"
# # tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# model_7.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])



# start training -- model_v8
mc = tf.keras.callbacks.ModelCheckpoint('/tfdepth/rss/model_v8/weights{epoch:08d}.h5', save_weights_only=False, period=5)
# NAME = "depth_net_2.0"
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
model_8.fit(train_gen, steps_per_epoch =1500, validation_data = validation_gen, epochs=100, validation_steps= 100, callbacks=[mc])