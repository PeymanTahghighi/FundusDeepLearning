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
        x = tf.expand_dims(x, 0)
        # 224 224
        x = tf.layers.conv2d(x, 64, (3, 3), strides=1, activation='relu',padding='same')
        x = tf.layers.conv2d(x, 64, (3, 3), strides=1, activation='relu',padding='same')
        x0 = x
        x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 112 112
        x = tf.layers.conv2d(x, 128, (3, 3), strides=1, activation='relu',padding='same')
        x = tf.layers.conv2d(x, 128, (3, 3), strides=1, activation='relu',padding='same')
        x1 = x
        x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 56 56
        x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation='relu',padding='same');
        x2 = x
        x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 28 28
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x3 = x
        x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 14 14
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x = tf.layers.conv2d(x, 512, (3, 3), strides=1, activation='relu', padding='same');
        x4 = x
        x = tf.layers.max_pooling2d(x,pool_size=(2,2),strides=(2,2));
        # 7 7

        self.placeholders.update({'img_feat': [tf.squeeze(x0), tf.squeeze(x1), tf.squeeze(x2)]})
        #print(self.placeholders['img_feat']);




