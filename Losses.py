import tensorflow as tf
import Config

def EMD(pred,placeholders):
    with tf.variable_scope("EMD"):
        gtPoints = placeholders['labels'];
        total = 0.0;
        total = tf.abs(tf.subtract(pred,gtPoints,name="_subtract"),name="_abs");
        total = tf.reduce_mean(total,name="_mean");
    return total;
