import tensorflow as tf
import numpy as np
import Config

def EMD(pred,placeholders):
    with tf.variable_scope("EMD"):
        gtPoints = placeholders['labels'];
        #gtPoints = tf.squeeze(gtPoints);
        #pred = tf.squeeze(pred);
        total = tf.abs(tf.subtract(pred, gtPoints, name="_subtract"));
        #ssim = (tf.reduce_mean(tf.image.ssim(pred,gtPoints,max_val=1.0,filter_size=11,k1=0.1),axis=0));
        mult = tf.divide(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.multiply(pred,gtPoints), name="_sum", axis=3)
                                                            , axis=2), axis=1), axis=0),
                  y=Config.NETWORK_SIZE * Config.NETWORK_SIZE * placeholders['dataSize']);

        #gtPoints = tf.math.log(gtPoints,name="_logNatrual");
        diff = tf.pow( tf.subtract(pred, gtPoints, name="_subtract"),2);
        #diffPow = tf.pow(diff,2,name="_power");

        loss = tf.divide(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(diff, name="_sum", axis=3)
                                                                   , axis=2), axis=1), axis=0),
                         y=Config.NETWORK_SIZE * Config.NETWORK_SIZE * placeholders['dataSize']);
        loss = loss + (1-tf.multiply(mult,y=0.5));
        thresh = tf.reduce_sum(total,axis = 3,keep_dims=True)
        thresh = tf.math.less_equal(thresh,0.05);
        acc = tf.math.count_nonzero(thresh);
        acc = tf.divide(acc,Config.NETWORK_SIZE*Config.NETWORK_SIZE*tf.cast(placeholders["dataSize"],dtype=tf.int64));
        #loss = tf.divide(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(total,name="_sum",axis = 3)
                                                  #       ,axis = 2),axis=1),axis=0),
                        # y=Config.NETWORK_SIZE*Config.NETWORK_SIZE*placeholders['dataSize']);
        #loss = tf.reduce_mean(total,name="_mean",axis=1);
        return loss,acc,thresh;
