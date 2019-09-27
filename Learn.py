import tensorflow as tf
import numpy as np

x = tf.get_variable("x_scalar",shape = [],initializer=tf.random_normal_initializer());
summary = tf.summary.scalar(name = "summary",tensor=x);

init = tf.global_variables_initializer();

wit


