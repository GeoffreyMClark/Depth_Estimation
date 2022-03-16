import pandas as pd
import tensorflow as tf
import logging
# tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, AveragePooling2D
import dataset_prep
# import depth_prediction_net
import loss

import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


width = 160
height = 90


# get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('./model_v4/weights00000100.h5',custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})


img_list = sorted(os.listdir('./test/'))

for i in range (len(img_list)):
    rgb = cv2.imread("./test/"+img_list[i])
    rgb = cv2.resize(rgb, (width, height))
    # Prediction
    rgb = np.array(rgb).reshape(-1, height, width, 3)
    rgb = np.float32(rgb / 255.)
    _, _, pred = model.predict(rgb)
    pred = pred[0, :, :, 0]
    np.save('./test/pred-'+str(i)+'.npy',pred)
    pred = np.uint8(255.*pred)
    cv2.imwrite('./test/pred-'+str(i)+'.jpg',pred)
