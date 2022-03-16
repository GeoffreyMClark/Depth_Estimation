import pandas as pd
import loss
import tensorflow.compat.v1 as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

TF_FORCE_GPU_ALLOW_GROWTH=True


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import tensorflow.compat.v1
tensorflow.compat.v1.disable_v2_behavior()


from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
get_loss = loss.get_loss()


model_fname = './model_v4/weights00000100.h5'


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

model = load_model(model_fname, custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, './', 'depth_model.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, './', 'depth_model.pb', as_text=False)

# from tensorflow.python.platform import gfile

# f = gfile.FastGFile("xor.pb", 'rb')
# graph_def = tf.GraphDef()
# # Parses a serialized binary message into the current message.
# graph_def.ParseFromString(f.read())
# f.close()

# sess.graph.as_default()
# # Import a serialized TensorFlow `GraphDef` protocol buffer
# # and place into the current default `Graph`.
# tf.import_graph_def(graph_def)

# [<tf.Tensor 'cropping2d/strided_slice:0' shape=(?, 23, 40, 1) dtype=float32>, <tf.Tensor 'cropping2d_2/strided_slice:0' shape=(?, 45, 80, 1) dtype=float32>, <tf.Tensor 'activation_20/Sigmoid:0' shape=(?, 90, 160, 1) dtype=float32>]
# [<tf.Tensor 'input_2:0' shape=(?, 90, 160, 3) dtype=float32>]
