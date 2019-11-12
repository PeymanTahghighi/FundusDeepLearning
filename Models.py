import tensorflow as tf
from Layers import *
from Losses import *
import Config
import os


class Model(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__.lower();

        self.vars = {};
        self.loss = 0;
        self.optimizer = None;
        self.layers = [];
        self.activations = [];
        self.inputs = None
        self.model = [];

    def _build(self):
        raise NotImplementedError;

    def build(self):
        self._build();
        self._loss();
        self.opt_op = self.optimizer.minimize(self.loss);

    def _loss(self):
        pass

    def predict(self):
        pass

    def save(self, sess):
        if not sess:
            raise AttributeError("Session not provided");

        #saver = tf.train.Saver()

        saver = tf.train.Saver(self.vars);
        savePath = os.path.sep.join([Config.BASE_MODEL_PATH, self.name, ".mdl"]);
        saver.save(sess, save_path=savePath);
        print("Model saved in file :{}".format(savePath));


class GCN(Model):
    def __init__(self, placeholders, **kwargs):
        super(GCN,self).__init__(**kwargs);
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.LEARNING_RATE);
        self.placeholders = placeholders;
        self.inputs = placeholders['features']

        self.build();
        self.graphNetwork = placeholders['graphNetwork'];

    def weight_variable(self,shape):
        init = tf.random_normal_initializer(stddev=0.02);
        return tf.get_variable(name="W", initializer=init, shape=shape, dtype=tf.float32);

    def bias_variable(self,shape):
        init = tf.random_normal_initializer(stddev=0.02);
        return tf.get_variable(name="B", initializer=init, shape=shape, dtype=tf.float32);

    def max_pool(self,x, ksize, stride, name):
        """
        Create a max pooling layer
        :param x: input to max-pooling layer
        :param ksize: size of the max-pooling filter
        :param stride: stride of the max-pooling filter
        :param name: layer name
        :return: The output array
        """
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding="SAME",
                              name=name)

    def conv_layer(self,x, filter_size, num_filters, stride, name):
        with tf.variable_scope(name):
            # x = tf.expand_dims(x, 0)
            num_in_channel = x.get_shape().as_list()[-1];
            shape = [filter_size, filter_size, num_in_channel, num_filters];
            W = self.weight_variable(shape);

            B = self.bias_variable(shape=[num_filters]);

            # tf.summary.histogram("weight",W);

            layer = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME");
            layer += B;
            return tf.nn.relu(layer);

    def _build(self):
        self.buildImageNetwork();
        self.layers.append(GraphProjection(placeholders=self.placeholders));
        self.layers.append(GraphConvolution(input_dim=Config.IMAGE_FEATURE_DIM,
                                            output_dim=Config.FEATURES_HIDDEN,
                           placeholders = self.placeholders));
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=Config.FEATURES_HIDDEN,
                                                output_dim=Config.FEATURES_HIDDEN,
                                                placeholders=self.placeholders));
        self.layers.append(GraphConvolution(input_dim=Config.FEATURES_HIDDEN,output_dim=1,
                                            placeholders=self.placeholders));




    def _loss(self):
        eltwise = [3, 5, 7,9,11]
        self.activations.append(self.inputs)
        for idx, layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            self.activations.append(hidden);

        out = self.activations[-1];
        self.loss = EMD(out,self.placeholders);

    def buildImageNetwork(self):
        x = self.placeholders['imageInput']
        #x = tf.expand_dims(x, 0)
        stride = 1;
        conv1_1 = self.conv_layer(x, 3, 64, stride, "conv1_1");
        conv1_2 = self.conv_layer(conv1_1, 3, 64, stride, "conv1_2");
        pool1 = self.max_pool(conv1_2, ksize=2, stride=2, name="pool1");
        pool1bn = tf.layers.batch_normalization(pool1);

        conv2_1 = self.conv_layer(pool1bn, 3, 128, stride, "conv2_1");
        conv2_2 = self.conv_layer(conv2_1, 3, 128, stride, "conv2_2");
        pool2 = self.max_pool(conv2_2, ksize=2, stride=2, name="pool2");
        pool2bn = tf.layers.batch_normalization(pool2);

        conv3_1 = self.conv_layer(pool2bn, 3, 256, stride, "conv3_1");
        conv3_2 = self.conv_layer(conv3_1, 3, 256, stride, "conv3_2");
        conv3_3 = self.conv_layer(conv3_2, 3, 256, stride, "conv3_3");
        pool3 = self.max_pool(conv3_3, ksize=2, stride=2, name="pool3");
        pool3bn = tf.layers.batch_normalization(pool3);

        conv4_1 = self.conv_layer(pool3bn, 3, 512, stride, "conv4_1");
        conv4_2 = self.conv_layer(conv4_1, 3, 512, stride, "conv4_2");
        conv4_3 = self.conv_layer(conv4_2, 3, 512, stride, "conv4_3");
        pool4 = self.max_pool(conv4_3, ksize=2, stride=2, name="pool4");
        pool4bn = tf.layers.batch_normalization(pool4);

        conv5_1 = self.conv_layer(pool4bn, 3, 512, stride, "conv5_1");
        conv5_2 = self.conv_layer(conv5_1, 3, 512, stride, "conv5_2");
        conv5_3 = self.conv_layer(conv5_2, 3, 512, stride, "conv5_3");
        pool5 = self.max_pool(conv5_3, ksize=2, stride=2, name="pool5");
        pool5bn = tf.layers.batch_normalization(pool5);

        # 224 224
        # x = tf.layers.conv2d(x, 64, (3, 3), strides=1, activation='relu',padding='same')
        # x = tf.layers.conv2d(x, 64, (3, 3), strides=1, activation='relu',padding='same')
        # x0 = x
        # x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # # 112 112
        # x = tf.layers.conv2d(x, 128, (3, 3), strides=1, activation='relu',padding='same')
        # x = tf.layers.conv2d(x, 128, (3, 3), strides=1, activation='relu',padding='same')
        # x1 = x
        # x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # # 56 56
        # x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        # x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        # x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        # x2 = x
        # x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # # 28 28
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x3 = x
        # x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # # 14 14
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        # x4 = x
        # x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 7 7

        self.placeholders.update({'img_feat': [tf.squeeze(pool1), tf.squeeze(pool2), tf.squeeze(pool3)]})
        #print(self.placeholders['img_feat']);




