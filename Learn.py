# import tensorflow as tf
# import numpy as np
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import Fetcher
# import math
#
# from tensorflow.examples.tutorials.mnist import input_data
#
# img_h = img_w = 100  # MNIST images are 28x28
# img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
# n_classes = 100  # Number of classes, one class per digit
# n_channels = 3
# batch_size = 50 # Training batch size
#
#
# def randomize(x, y):
#     permutation = np.random.permutation(y.shape[0]);
#     shuffled_x = x[permutation, :, :, :];
#     shuffled_y = y[permutation];
#     return shuffled_x, shuffled_y;
#
#
# def get_next_batch(x, y, start, end):
#     x_batch = x[start:end];
#     y_batch = y[start:end];
#     return x_batch, y_batch;
#
#
# # Helper function for variables and network layers
# def weight_variable(shape):
#     init = tf.random_normal_initializer(stddev=0.01);
#     return tf.get_variable(name="W", initializer=init, shape=shape, dtype=tf.float32);
#
#
# def bias_variable(shape):
#     init = tf.random_normal_initializer(stddev=0.01);
#     return tf.get_variable(name="B", initializer=init, shape=shape, dtype=tf.float32);
#
#
# def extractFeatures(featuresList = []):
#     agg = [];
#     for i in range(len(featuresList)):
#         shape = featuresList[i].get_shape().as_list();
#         x1 = shape[1] / 100 * 50;
#
#         coordsfloorStacked = np.zeros((batch_size, 3), dtype=np.float);
#         coordsceilStacked = np.zeros((batch_size, 3), dtype=np.float);
#         coordsfcStacked = np.zeros((batch_size, 3), dtype=np.float);
#         coordscfStacked = np.zeros((batch_size, 3), dtype=np.float);
#
#         for j in range(batch_size):
#             coordsfloorStacked[j] = (j,math.floor(x1),math.floor(x1));
#             coordsceilStacked[j] = (j, math.ceil(x1), math.ceil(x1));
#             coordsfcStacked[j] = (j, math.floor(x1), math.ceil(x1));
#             coordscfStacked[j] = (j, math.ceil(x1), math.floor(x1));
#
#         coordstffloortf = tf.convert_to_tensor(value=coordsfloorStacked, dtype=tf.float32);
#         coordstfceiltf = tf.convert_to_tensor(value=coordsceilStacked, dtype=tf.float32);
#         coordstfcftf = tf.convert_to_tensor(value=coordscfStacked, dtype=tf.float32);
#         coordstffctf = tf.convert_to_tensor(value=coordsfcStacked, dtype=tf.float32);
#
#         coordstffloortf = tf.cast(coordstffloortf, dtype=tf.int32);
#         coordstfceiltf = tf.cast(coordstfceiltf, dtype=tf.int32);
#         coordstfcftf = tf.cast(coordstfcftf, dtype=tf.int32);
#         coordstffctf = tf.cast(coordstffctf, dtype=tf.int32);
#
#         v133 = tf.gather_nd(featuresList[i], [coordstffloortf], name="_gather_1{}3".format(i));
#         v233 = tf.gather_nd(featuresList[i], [coordstfceiltf], name="_gather_2{}3".format(i));
#         v333 = tf.gather_nd(featuresList[i], [coordstfcftf], name="_gather_3{}3".format(i));
#         v433 = tf.gather_nd(featuresList[i], [coordstffctf], name="_gather_4{}3".format(i));
#
#         final = tf.divide(tf.add_n([v133, v233, v333, v433]), y=tf.constant(value=4.0, dtype=tf.float32));
#         final = tf.squeeze(final);
#         agg.append(final);
#     # ----------------------------------------------------------------------------
#
#     final = tf.squeeze(tf.concat(agg, axis=1), name="_finalFeatures");
#     s = final.get_shape().as_list();
#
#     return final,s[1];
#     # ---------------------------------------------------------
#
#
# def conv_layer(x, filter_size, num_filters, stride, name):
#     with tf.variable_scope(name):
#         # x = tf.expand_dims(x, 0)
#         num_in_channel = x.get_shape().as_list()[-1];
#         shape = [filter_size, filter_size, num_in_channel, num_filters];
#         W = weight_variable(shape);
#
#         B = bias_variable(shape=[num_filters]);
#
#         # tf.summary.histogram("weight",W);
#
#         layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME");
#         layer += B;
#         return tf.nn.relu(layer);
#
#
#
# def max_pool(x, ksize, stride, name):
#     """
#     Create a max pooling layer
#     :param x: input to max-pooling layer
#     :param ksize: size of the max-pooling filter
#     :param stride: stride of the max-pooling filter
#     :param name: layer name
#     :return: The output array
#     """
#     return tf.nn.max_pool(x,
#                           ksize=[1, ksize, ksize, 1],
#                           strides=[1, stride, stride, 1],
#                           padding="SAME",
#                           name=name)
#
#
# def flatten_layer(layer):
#     """
#     Flattens the output of the convolutional layer to be fed into fully-connected layer
#     :param layer: input array
#     :return: flattened array
#     """
#     with tf.variable_scope('Flatten_layer'):
#         layer_shape = layer.get_shape()
#         num_features = layer_shape[1:4].num_elements()
#         layer_flat = tf.reshape(layer, [-1, num_features])
#     return layer_flat
#
#
# def fc_layer(x, num_units, name, use_relu=True):
#     """
#     Create a fully-connected layer
#     :param x: input from previous layer
#     :param num_units: number of hidden units in the fully-connected layer
#     :param name: layer name
#     :param use_relu: boolean to add ReLU non-linearity (or not)
#     :return: The output array
#     """
#     with tf.variable_scope(name):
#         in_dim = x.get_shape()[1]
#         W = weight_variable(shape=[in_dim, num_units])
#         # tf.summary.histogram('weight', W)
#         b = bias_variable(shape=[num_units])
#         # tf.summary.histogram('bias', b)
#         layer = tf.matmul(x, W)
#         layer += b
#         #if use_relu:
#             #layer = tf.nn.relu(layer)
#         return layer
#
#
# # --------------------------------------------------------------
# data = Fetcher.DataFetcher("Dataset\heightmap", "Dataset\image");
# data.load();
# # x_train, y_train, x_valid, y_valid = load_data(mode="train");
#
# # Hyperparameters
# logs_path = "./logs_dir";
# lr = 3e-4  # The optimization initial learning rate
# epochs = 200  # Total number of training epochs
#
# display_freq = 100  # Frequency of displaying the training results
# stride = 1;
# h1 = 4096;
# h2 = 1000;
# global_step = 0;
# decay = 0.08;
#
# lrschedule = tf.compat.v1.train.exponential_decay(
#     learning_rate=lr,
#     global_step=global_step,
#     decay_rate=decay,
#     decay_steps=100
# )
#
#
# def loss(labels, pred):
#     total = tf.reduce_mean(tf.abs(tf.subtract(labels, pred)));
#     return total;
#
#
# def accuracy(labels, pred):
#     diff = tf.abs(tf.subtract(labels, pred));
#     threshold = tf.constant(value=0.1, dtype=tf.float32, name="_Threshold");
#     all = tf.constant(value=batch_size, dtype=tf.float32, name="_size");
#     thresh = tf.cast(tf.cast(tf.less_equal(diff, threshold), dtype=tf.int32), dtype=tf.float32);
#     acc = tf.divide(tf.reduce_sum(thresh, axis=0), all);
#     return acc;
#
#
# with tf.variable_scope("Input"):
#     x = tf.placeholder(tf.float32, shape=[None, img_h, img_w, n_channels], name="X");
#     y = tf.placeholder(tf.float32, shape=[None, 1], name="Y");
#     # x = tf.expand_dims(x, 0);
#     conv1_1 = conv_layer(x, 3, 64, stride, "conv1_1");
#     conv1_2 = conv_layer(conv1_1, 3, 64, stride, "conv1_2");
#     pool1 = max_pool(conv1_2, ksize=2, stride=2, name="pool1");
#     pool1bn = tf.layers.batch_normalization(pool1);
#
#     conv2_1 = conv_layer(pool1bn, 3, 128, stride, "conv2_1");
#     conv2_2 = conv_layer(conv2_1, 3, 128, stride, "conv2_2");
#     pool2 = max_pool(conv2_2, ksize=2, stride=2, name="pool2");
#     pool2bn = tf.layers.batch_normalization(pool2);
#
#     conv3_1 = conv_layer(pool2bn, 3, 256, stride, "conv3_1");
#     conv3_2 = conv_layer(conv3_1, 3, 256, stride, "conv3_2");
#     conv3_3 = conv_layer(conv3_2, 3, 256, stride, "conv3_3");
#     pool3 = max_pool(conv3_3, ksize=2, stride=2, name="pool3");
#     pool3bn = tf.layers.batch_normalization(pool3);
#
#     conv4_1 = conv_layer(pool3bn, 3, 512, stride, "conv4_1");
#     conv4_2 = conv_layer(conv4_1, 3, 512, stride, "conv4_2");
#     conv4_3 = conv_layer(conv4_2, 3, 512, stride, "conv4_3");
#     pool4 = max_pool(conv4_3, ksize=2, stride=2, name="pool4");
#     pool4bn = tf.layers.batch_normalization(pool4);
#
#     conv5_1 = conv_layer(pool4bn, 3, 512, stride, "conv5_1");
#     conv5_2 = conv_layer(conv5_1, 3, 512, stride, "conv5_2");
#     conv5_3 = conv_layer(conv5_2, 3, 512, stride, "conv5_3");
#     pool5 = max_pool(conv5_3, ksize=2, stride=2, name="pool5");
#     pool5bn = tf.layers.batch_normalization(pool5);
#
#     feature,size = extractFeatures([conv1_2,conv2_2, conv3_3, conv4_3]);
#     f = fc_layer(feature, size, "FC0");
#
#     #for i in range(8):
#      #   f = fc_layer(f,896-i*100,"FC" + str(i+1));
#
#     out = fc_layer(f,1,"FCF");
#
#     # layer_flat = flatten_layer(pool5bn);
#     # fc1 = fc_layer(layer_flat, h1, "FC1", True);
#     # out = fc_layer(fc1, 1, "OUT", False);
#     # out = tf.squeeze(out);
#
# with tf.variable_scope("Train"):
#     with tf.variable_scope("Loss"):
#         # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits = out),name = "Loss");
#         loss = loss(labels=y, pred=out);
#         tf.summary.scalar("loss", loss);
#     with tf.variable_scope("Optimizer"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=lrschedule, name="Adam-OP").minimize(loss);
#
#     with tf.variable_scope("Accuracy"):
#         correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1), name='correct_pred')
#         # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#         accuracy = accuracy(labels=y, pred=out);
#         # tf.summary.scalar('accuracy', accuracy)
#     with tf.variable_scope('Prediction'):
#         cls_prediction = tf.argmax(out, axis=1, name='predictions')
#
#     init = tf.global_variables_initializer();
#     summary = tf.summary.merge_all();
#
#     learning_rate = tf.placeholder(tf.float32, shape=[])
#
#     sess = tf.InteractiveSession();
#     sess.run(init);
#
#     summary_writer = tf.summary.FileWriter(logs_path, sess.graph);
#     saver = tf.train.Saver();
#     num_iter = data.getTrainDataSize();
#     filename = "{}-{}-{}.txt".format(epochs,lr,decay);
#     file = open(filename,"w+");
#     file.write("Epochs :{} \t Learning rate : {} \t Decay : {} \n".format(epochs,lr,decay));
#     for epoch in range(epochs):
#         print("Training epoch : {}".format(epoch + 1));
#         all_loss = np.zeros(int((num_iter / batch_size)), dtype='float32');
#         all_acc = np.zeros(int((num_iter / batch_size)), dtype='float32');
#         # x_train,y_train = randomize(x_train,y_train);
#         for iteration in range(int((num_iter / batch_size))):
#             global_step += 1;
#             start = iteration * batch_size;
#             end = (iteration + 1) * batch_size;
#             img, rad = data.fetchTrainRadius(batchSize=batch_size);
#             # x_batch,y_batch = get_next_batch(x_train,y_train,start,end);
#             # img = tf.expand_dims(img, 0)
#             # img = np.expand_dims(img,0);
#             # img = tf.cast(img,dtype=tf.float32);
#             # img = img.eval(sess = sess);
#             # img.numpy();
#             # if(epoch > 50):
#             #   lr=0.1e-5;
#             feed_dict = {x: img, y: rad};
#
#             _, loss_batch, acc_batch,ret,f = sess.run([optimizer, loss, accuracy,out,feature], feed_dict=feed_dict);
#             all_loss[iteration] = loss_batch;
#             # all_acc[iteration] = acc_batch;
#             mean_loss = np.mean(all_loss[np.where(all_loss)]);
#             # mean_acc = np.mean(all_acc[np.where(all_acc)]);
#             # print("Loss={}\tacc={}".format(loss_batch,acc_batch));
#             # if iteration % display_freq ==0:
#             # loss_batch,acc_batch,summary_tr = sess.run([loss,accuracy,summary],feed_dict=feed_dict);
#             # summary_writer.add_summary(summary_tr,global_step);
#             print("iter {}:\tlr = {}\t Loss={}:\tacc={}".
#                   format(iteration, lr, mean_loss, acc_batch))
#             file.write("iter {}:\tlr = {}\t Loss={}:\tacc={} \n".
#                   format(iteration, lr, mean_loss, acc_batch));
#         imgVal, radVal = data.fetchValidRadius(batchSize =50);
#         feed_dict_valid = {x:imgVal,y:radVal};
#         loss_valid,acc_valid = sess.run([loss,accuracy],feed_dict=feed_dict_valid);
#         print('---------------------------------------------------------')
#         print("Epoch: {}, validation loss: {}, validation accuracy: {}".
#         format(epoch + 1, loss_valid, acc_valid))
#         print('---------------------------------------------------------')
#     file.close();
#
#     save_path = saver.save(sess,"model.mdl",global_step=global_step);
#     print("Model Successfully saved at : " + save_path);
#
#     # Test
#     # x_test,y_test = load_data(mode="test");
#     # feed_dict = {x:x_test,y:y_test};
#     # loss_test,acc_test = sess.run([loss,accuracy],feed_dict=feed_dict);
#     # print('---------------------------------------------------------')
#     # print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
#     # print('---------------------------------------------------------')
#     #
#     # cls_pred = sess.run(cls_prediction, feed_dict=feed_dict)
#     # cls_true = np.argmax(y_test, axis=1)
#     # plot_images(x_test,cls_true,cls_pred,title="ssadas");
#     # plt.show();
#     # -------------------------------------------------------
#
# # placeholders = {
# #     'features': tf.placeholder(tf.float32, shape=(None, 2),name="vertexCoordinates"),
# # }
# #
# #
# #
# # sess = tf.Session()
# # x56 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft",shape=(10,10000,1),dtype=tf.float32);
# # x26 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft1",shape=(10,26,26,10),dtype=tf.float32);
# # x7 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft2",shape=(10,7,7,10),dtype=tf.float32);
# #
# #
# # coords = pickle.load(file=open("surface.dat","rb"));
# # sess.run(tf.global_variables_initializer());
# # print(sess.run(x56));
# # x56 = tf.squeeze(x56,2);
# # print(sess.run(x56));
# # #coordstf=tf.convert_to_tensor(value=coords,dtype=tf.float32);
# # #coordsstacked = np.zeros((10,10000,2),dtype=np.float);
# # #coordsstacked[0] = coordsstacked;
# # coordstf56=np.multiply(coords,55.0/100.0);
# #
# # coordstf56floorStacked = np.zeros((10,10000,3),dtype=np.float);
# # coordstf56ceilStacked = np.zeros((10,10000,3),dtype=np.float);
# # coordstf56fcStacked = np.zeros((10,10000,3),dtype=np.float);
# # coordstf56cfStacked = np.zeros((10,10000,3),dtype=np.float);
# #
# #
# # coordstf56floor = np.floor(coordstf56);
# # coordstf56ceil = np.ceil(coordstf56);
# # floorX = coordstf56floor[:,0];
# # ceilX = coordstf56ceil[:,0];
# #
# # floorY = coordstf56floor[:,1];
# # ceilY = coordstf56ceil[:,1];
# #
# # coordstf56fc = np.stack([floorX,ceilY],axis=1);
# # coordstf56cf = np.stack([ceilX,floorY],axis=1);
# # for j in range(10):
# #     for i in range(10000):
# #         coordstf56floorStacked[j][i] = [j,coordstf56floor[i][0],coordstf56floor[i][1]];
# #         coordstf56ceilStacked[j][i] = [j, coordstf56ceil[i][0], coordstf56ceil[i][1]];
# #         coordstf56fcStacked[j][i] = [j, coordstf56fc[i][0], coordstf56fc[i][1]];
# #         coordstf56cfStacked[j][i] = [j, coordstf56cf[i][0], coordstf56cf[i][1]];
# #
# # coordstf56floortf = tf.convert_to_tensor(value=coordstf56floorStacked,dtype=tf.float32);
# # coordstf56ceiltf= tf.convert_to_tensor(value=coordstf56ceilStacked,dtype=tf.float32);
# # coordstf56cftf=tf.convert_to_tensor(value=coordstf56cfStacked,dtype=tf.float32);
# # coordstf56fctf=tf.convert_to_tensor(value=coordstf56fcStacked,dtype=tf.float32);
# #
# #
# # coordstf56floortf = tf.cast(coordstf56floortf,dtype=tf.int32);
# # coordstf56ceiltf = tf.cast(coordstf56ceiltf,dtype=tf.int32);
# # coordstf56cftf = tf.cast(coordstf56cftf,dtype=tf.int32);
# # coordstf56fctf = tf.cast(coordstf56fctf,dtype=tf.int32);
# #
# #
# # v1 = tf.gather_nd(x56, [coordstf56floortf]);
# # v2 = tf.gather_nd(x56, [coordstf56ceiltf]);
# # v3 = tf.gather_nd(x56, [coordstf56cftf]);
# # v4 = tf.gather_nd(x56, [coordstf56fctf]);
# #
# # final = tf.divide(tf.add_n([v1,v2,v3,v4]),y=tf.constant(value=4.0,dtype=tf.float32));
# # final=tf.squeeze(final);
# # print(sess.run(final));
# #
# #
# #
# #
# #
# # coordstf26=tf.multiply(coordstf,tf.constant(value=(25.0/223.0),dtype=tf.float32));
# # coordstf7=tf.multiply(coordstf,tf.constant(value=(6.0/223.0),dtype=tf.float32));
# #
# # coordstf56 = tf.cast(coordstf56,dtype=tf.int32);
# # coordstf26 = tf.cast(coordstf26,dtype=tf.int32);
# # coordstf7 = tf.cast(coordstf7,dtype=tf.int32);
# # # X = coordstf[:, 0]
# # # Y = coordstf[:, 1]
# #
# # y1 = tf.gather_nd(x56, [coordstf56])
# # y2 = tf.gather_nd(x26, [coordstf26])
# # y3 = tf.gather_nd(x7, [coordstf7])
# # final = tf.concat([y1,y2,y3],axis=2);
# # print(sess.run(final))
#
# # import matplotlib.pyplot as plt
# #
# # import bezier
# #
# # nodes = np.asfortranarray([
# # [0.0, 0.5, 1.0 , 0.125, 0.375, 0.25],
# # [0.0, 0.0, 0.25, 0.5  , 0.375, 1.0 ],
# # [0.0, 0.0, 0.0, 0.0  , 0.0, 0.0 ],
# # ])
# #
# # surface = bezier.Surface(nodes, degree=3)
# # p = surface.evaluate_cartesian(0,0);
# #
# # print(p);
# #
# # surface.plot(pts_per_edge=256);
# # plt.show();
#
# # a = tf.get_variable(initializer=tf.constant(value=1,dtype=tf.float32,shape=(10,20)),dtype=tf.float32,name="koft");
# # b = tf.get_variable(initializer=tf.random_normal_initializer(),shape=(20),dtype=tf.float32,name="koft1");
# # add = a+b;
# #
# # sess = tf.Session();
# # sess.run(tf.global_variables_initializer());
# # print(sess.run(add));


# Demonstrates similarities between pcolor, pcolormesh, imshow and pcolorfast
# for drawing quadrilateral grids.


import matplotlib.pyplot as plt
import numpy as np

import sys
EPSILON = sys.float_info.epsilon  # Smallest possible difference.

def convert_to_rgb(minval, maxval, val, colors):
    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

if __name__ == '__main__':
    minval, maxval = 0, 450
    steps = 500
    delta = float(maxval-minval) / steps
    colors = [(0, 0, 0), (0, 35, 35),(0, 50, 0) ,(35, 35, 0),(50, 0, 0)]  # [BLUE, GREEN, RED]
    print('  Val       R    G    B')
    for i in range(steps+1):
        val = minval + i*delta
        r, g, b = convert_to_rgb(minval, maxval, val, colors)
        print('{:.3f} -> ({:3d}, {:3d}, {:3d})'.format(val, r, g, b))
