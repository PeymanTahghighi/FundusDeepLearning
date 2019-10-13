import tensorflow as tf
import numpy as np
import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# from tensorflow.examples.tutorials.mnist import input_data
#
#
# img_h = img_w = 28  # MNIST images are 28x28
# img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
# n_classes = 10  # Number of classes, one class per digit
# n_channels = 1
#
# def load_data(mode='train'):
#     """
#     Function to (download and) load the MNIST data
#     :param mode: train or test
#     :return: images and the corresponding labels
#     """
#     mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#     if mode == 'train':
#         x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
#                                              mnist.validation.images, mnist.validation.labels
#         x_train, _ = reformat(x_train, y_train)
#         x_valid, _ = reformat(x_valid, y_valid)
#         return x_train, y_train, x_valid, y_valid
#     elif mode == 'test':
#         x_test, y_test = mnist.test.images, mnist.test.labels
#         x_test, _ = reformat(x_test, y_test)
#     return x_test, y_test
#
#
# def reformat(x, y):
#     """
#     Reformats the data to the format acceptable for convolutional layers
#     :param x: input array
#     :param y: corresponding labels
#     :return: reshaped input and labels
#     """
#     img_size, num_ch, num_class = int(np.sqrt(x.shape[-1])), 1, len(np.unique(np.argmax(y, 1)))
#     dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
#     labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
#     return dataset, labels
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
# # Helper function for variables and network layers
# def weight_variable(shape):
#     init = tf.truncated_normal_initializer(stddev=0.01);
#     return tf.get_variable(name="W", initializer=init, shape=shape, dtype=tf.float32);
#
#
# def bias_variable(shape):
#     init = tf.truncated_normal_initializer(stddev=0.01);
#     return tf.get_variable(name="B", initializer=init, shape=shape, dtype=tf.float32);
#
#
# def conv_layer(x, filter_size, num_filters, stride, name):
#     with tf.variable_scope(name):
#         num_in_channel = x.get_shape().as_list()[-1];
#         shape = [filter_size, filter_size, num_in_channel, num_filters];
#         W = weight_variable(shape);
#
#         tf.summary.histogram("weight",W);
#
#         B = bias_variable(shape=[num_filters]);
#
#         tf.summary.histogram("bias",B);
#
#         layer = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME");
#
#         layer+=B;
#
#         return tf.nn.relu(layer);
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
#         tf.summary.histogram('weight', W)
#         b = bias_variable(shape=[num_units])
#         tf.summary.histogram('bias', b)
#         layer = tf.matmul(x, W)
#         layer += b
#         if use_relu:
#             layer = tf.nn.relu(layer)
#         return layer
#
# def plot_images(images,cls_true,cls_pred=None,title=None):
#     fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     for i, ax in enumerate(axes.flat):
#         # Plot image.
#         ax.imshow(np.squeeze(images[i]), cmap='binary')
#
#         # Show true and predicted classes.
#         if cls_pred is None:
#             ax_title = "True: {0}".format(cls_true[i])
#         else:
#             ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
#
#         ax.set_title(ax_title)
#
#         # Remove ticks from the plot.
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     if title:
#         plt.suptitle(title, size=20)
#     plt.show(block=False)
#
#
# # --------------------------------------------------------------
#
# x_train, y_train, x_valid, y_valid = load_data(mode="train");
#
# # Hyperparameters
# logs_path = "./logs_dir";
# lr = 0.001  # The optimization initial learning rate
# epochs = 10  # Total number of training epochs
# batch_size = 100  # Training batch size
# display_freq = 100  # Frequency of displaying the training results
#
# filter_size1 = 5;
# num_filters1 = 16;
# stride = 1;
#
# filter_size2 = 5;
# num_filters2 = 32;
#
# h1 = 128;
#
# with tf.variable_scope("Input"):
#     x = tf.placeholder(tf.float32,shape = [None,img_h,img_w,n_channels],name = "X");
#     y = tf.placeholder(tf.float32,shape=[None,n_classes],name="Y");
#     conv1 = conv_layer(x,filter_size1,num_filters1,stride,"conv1");
#     pool1 = max_pool(conv1,ksize=2,stride=2,name="pool1");
#     conv2 = conv_layer(pool1,filter_size2,num_filters2,stride,"conv2");
#     pool2 = max_pool(conv2,ksize=2,stride=2,name="pool2");
#     layer_flat = flatten_layer(pool2);
#     fc = fc_layer(layer_flat,h1,"FC1",True);
#     out = fc_layer(fc,n_classes,"OUT",False);
#
# with tf.variable_scope("Train"):
#     with tf.variable_scope("Loss"):
#         loss = tf.reduce_mean\
#             (tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = out),name = "Loss");
#         tf.summary.scalar("loss",loss);
#     with tf.variable_scope("Optimizer"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=lr,name="Adam-OP").minimize(loss);
#
#     with tf.variable_scope("Accuracy"):
#         correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1), name='correct_pred')
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
#         tf.summary.scalar('accuracy', accuracy)
#     with tf.variable_scope('Prediction'):
#         cls_prediction = tf.argmax(out, axis=1, name='predictions')
#
#     init = tf.global_variables_initializer();
#     summary = tf.summary.merge_all();
#
#     sess = tf.InteractiveSession();
#     sess.run(init);
#     global_step = 0;
#     summary_writer = tf.summary.FileWriter(logs_path,sess.graph);
#
#     num_iter = int(len(y_train)/batch_size);
#     for epoch in range(epochs):
#         print("Training epoch : {}".format(epoch+1));
#         x_train,y_train = randomize(x_train,y_train);
#         for iteration in range(num_iter):
#             global_step += 1;
#             start = iteration*batch_size;
#             end = (iteration + 1) *batch_size;
#
#             x_batch,y_batch = get_next_batch(x_train,y_train,start,end);
#
#             feed_dict = {x: x_batch,y: y_batch};
#             sess.run(optimizer,feed_dict=feed_dict);
#
#             if iteration % display_freq ==0:
#                 loss_batch,acc_batch,summary_tr = sess.run([loss,accuracy,summary],feed_dict=feed_dict);
#                 summary_writer.add_summary(summary_tr,global_step);
#                 print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
#                       format(iteration, loss_batch, acc_batch))
#
#         feed_dict_valid = {x:x_valid,y:y_valid};
#         loss_valid,acc_valid = sess.run([loss,accuracy],feed_dict=feed_dict_valid);
#         print('---------------------------------------------------------')
#         print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
#               format(epoch + 1, loss_valid, acc_valid))
#         print('---------------------------------------------------------')
#
#
#     #Test
#     x_test,y_test = load_data(mode="test");
#     feed_dict = {x:x_test,y:y_test};
#     loss_test,acc_test = sess.run([loss,accuracy],feed_dict=feed_dict);
#     print('---------------------------------------------------------')
#     print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
#     print('---------------------------------------------------------')
#
#     cls_pred = sess.run(cls_prediction, feed_dict=feed_dict)
#     cls_true = np.argmax(y_test, axis=1)
#     plot_images(x_test,cls_true,cls_pred,title="ssadas");
#     plt.show();
#     #-------------------------------------------------------

placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 2),name="vertexCoordinates"),
}



sess = tf.Session()
x56 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft",shape=(56,56,10),dtype=tf.float32);
x26 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft1",shape=(26,26,10),dtype=tf.float32);
x7 = tf.get_variable(initializer=tf.random_normal_initializer(),name="koft2",shape=(7,7,10),dtype=tf.float32);


coords = pickle.load(file=open("surface.dat","rb"));
sess.run(tf.global_variables_initializer());
coordstf=tf.convert_to_tensor(value=coords,dtype=tf.float32);
coordstf56=tf.multiply(coordstf,tf.constant(value=(55.0/223.0),dtype=tf.float32));
coordstf56floor = tf.floor(coordstf56);
coordstf56ceil = tf.ceil(coordstf56);
floorX = coordstf56floor[:,0];
ceilX = coordstf56ceil[:,0];

floorY = coordstf56floor[:,1];
ceilY = coordstf56ceil[:,1];

coordstf56fc = tf.stack([floorX,ceilY],axis=1);
coordstf56cf = tf.stack([ceilX,floorY],axis=1);

coordstf56floor = tf.cast(coordstf56floor,dtype=tf.int32);
coordstf56ceil = tf.cast(coordstf56ceil,dtype=tf.int32);
coordstf56cf = tf.cast(coordstf56cf,dtype=tf.int32);
coordstf56fc = tf.cast(coordstf56fc,dtype=tf.int32);

v1 = tf.gather_nd(x56, [coordstf56floor]);
v2 = tf.gather_nd(x56, [coordstf56ceil]);
v3 = tf.gather_nd(x56, [coordstf56cf]);
v4 = tf.gather_nd(x56, [coordstf56fc]);

final = tf.divide(tf.add_n([v1,v2,v3,v4]),y=tf.constant(value=4.0,dtype=tf.float32));
final=tf.squeeze(final);


print(sess.run(final))


coordstf26=tf.multiply(coordstf,tf.constant(value=(25.0/223.0),dtype=tf.float32));
coordstf7=tf.multiply(coordstf,tf.constant(value=(6.0/223.0),dtype=tf.float32));

coordstf56 = tf.cast(coordstf56,dtype=tf.int32);
coordstf26 = tf.cast(coordstf26,dtype=tf.int32);
coordstf7 = tf.cast(coordstf7,dtype=tf.int32);
# X = coordstf[:, 0]
# Y = coordstf[:, 1]

y1 = tf.gather_nd(x56, [coordstf56])
y2 = tf.gather_nd(x26, [coordstf26])
y3 = tf.gather_nd(x7, [coordstf7])
final = tf.concat([y1,y2,y3],axis=2);
print(sess.run(final))

