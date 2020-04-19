import os
import tensorflow as tf
BASE_PATH = "dataset";
TRAIN = "training";
VAL = "validation"
TEST = "evaluation";

BASE_FEATURE_OUTPUT = "Features";
BASE_MODEL_PATH = "Models"

MAX_WIDTH = 224;
MAX_HEIGHT = 224;
BATCH_SIZE = 12;

NETWORK_SIZE = 128;

IMAGE_FEATURE_DIM = 1280;
FEATURES_HIDDEN = 500;
IMG_WIDTH = 128;
IMG_HEIGHT = 128;

LEARNING_RATE = 1e-1;
EPOCHS = 25;
DROPOUT_RATIO = 0.25;
IS_TRAINING = tf.placeholder(tf.bool);
IS_TRAININGDO = tf.placeholder(tf.bool);


