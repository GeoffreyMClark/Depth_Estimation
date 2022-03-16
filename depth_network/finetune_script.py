import pandas as pd
import dataset_prep
import depth_prediction_net
import loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization, Concatenate
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorflow import keras

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
opt = Adam(lr=1e-5)

# Recreate the exact same model, including its weights and the optimizer
base_model = tf.keras.models.load_model('./model_v4/weights00000100.h5',custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})

fine_tune_autoencoder = models.Model(base_model.inputs, outputs=base_model.outputs)

fine_tune_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss,loss_weights= [1/64, 1/32, 1/16, 1/8, 1/4, 1])


# set a low learning rate
optimizer=keras.optimizers.Adam(1e-7)
epochs = 30
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step in range (1000):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            train, depth, mask = get_dataset.select_batch_mask()
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            # print("-----------check1---------------------")
            # print(depth[0].shape)
            train_1 = tf.reshape(train[0], [1, height, width, 3])
            train_2 = tf.reshape(train[1], [1, height, width, 3])

            # print(train_1.shape)
            depth_1 = tf.reshape(depth[0], [1, height, width, 1])
            depth_2 = tf.reshape(depth[1], [1, height, width, 1])
            mask_1 = tf.reshape(mask[0], [1, height, width, 1])
            mask_2 = tf.reshape(mask[1], [1, height, width, 1])

            logits_1 = fine_tune_autoencoder(train_1, training=True)
            logits_2 = fine_tune_autoencoder(train_2, training=True)
            # print("-----------check2---------------------")
            # Compute the loss value for this minibatch.
            loss_value = get_loss_finetune.autoencoder_loss(logits_1,logits_2,depth_1,depth_2,mask_1,mask_2)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, fine_tune_autoencoder.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, fine_tune_autoencoder.trainable_weights))

        # Log every 100 batches.
        if step % 100 == 0:
            print("Training loss (for one batch): ",loss_value)
            print(loss_value.shape)
            print("Seen so far: %s samples" % ((step + 1) * 2))
    if (epoch+1) % 10 == 0:        
        fine_tune_autoencoder.save('/tfdepth/rss/model_v4/weights00000100.h5')
        print("model has been saved")