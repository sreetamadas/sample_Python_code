# get tf efficientdet

# opening tensorboard in notebooks
#https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tensorboard_in_notebooks.ipynb#scrollTo=KBHp6M_zgjp4

# My TensorBoard isn't showing any data! What's wrong?
# https://github.com/tensorflow/tensorboard/blob/master/README.md

# Tensorboard cant see tfevent file generated using Intel AI academy Tensorflow scripts
# https://github.com/tensorflow/tensorboard/issues/1756


import os
os.getcwd()


#!pip install tensorboard

import tensorflow as tf; print(tf.__version__)

from tensorboard import version; print(version.VERSION)

%load_ext tensorboard

%tensorboard --logdir tf_run4   # this code is run just outside the logdir names tf_run4; logdir contains train & eval folders, as well as pipeline.config



%reload_ext tensorboard
%tensorboard --logdir tf_run4


