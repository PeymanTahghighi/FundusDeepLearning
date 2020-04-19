import tensorflow as tf
from Layers import *
from Losses import *
import Config
from tensorflow.contrib.layers import batch_norm, flatten,xavier_initializer_conv2d
from tensorflow.contrib.framework import arg_scope
import tensorflow.contrib.layers as tcl
from tflearn.layers.conv import global_avg_pool
import os


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.vars = {};
        self.loss = 0;
        self.optimizer = None;
        self.activations = [];
        self.inputs = None
        self.model = [];
        self.weightDecay = 0.0;
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _build(self):
        raise NotImplementedError;

    def build(self):
        self._build();
        self._loss();
        #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            lrschedule = tf.compat.v1.train.exponential_decay(
            learning_rate=Config.LEARNING_RATE,
            global_step=self.global_step,
            decay_rate=0.9,
            decay_steps=909 * 25
            , staircase=True,
            )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lrschedule);
            self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step);

    def _loss(self):
        pass

    def predict(self):
        pass

    def save(self, sess):
        if not sess:
            raise AttributeError("Session not provided");

        # saver = tf.train.Saver()

        saver = tf.train.Saver(self.vars);
        savePath = os.path.sep.join([Config.BASE_MODEL_PATH, self.name, ".mdl"]);
        saver.save(sess, save_path=savePath);
        print("Model saved in file :{}".format(savePath));


class GCN(Model):
    def __init__(self, placeholders, **kwargs):
        super(GCN, self).__init__(**kwargs);


        self.placeholders = placeholders;
        # self.inputs = placeholders['features']
        self.pred = None;
        self.features = None;
        self.accuracy = None;
        self.thresh = None;
        self.growthRate = 8;

        self.build();

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss);
            tf.summary.scalar("accuracy", self.accuracy);
            self.summary_op = tf.summary.merge_all();

    def weight_variable(self, shape, stddev):
        init = tf.initializers.random_normal(stddev=stddev);
        return tf.get_variable(name="W", initializer=init, shape=shape, dtype=tf.float32);

    def bias_variable(self, shape, stddev):
        init = tf.random_normal_initializer(stddev=stddev);
        return tf.get_variable(name="B", initializer=init, shape=shape, dtype=tf.float32);

    def max_pool(self, x, ksize, stride,name,padding="SAME"):
        return tf.nn.max_pool(x,
                              ksize=[ksize, ksize],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=name)

    def avg_pool(self, x, ksize, stride, name,padding="SAME"):
        return tf.nn.avg_pool2d(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=name)

    def Batch_Normalization(self,x, training, name):
        with arg_scope([batch_norm],
                       scope=name,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return tf.cond(training,
                           lambda: batch_norm(inputs=x,is_training=training, reuse=None),
                           lambda: batch_norm(inputs=x, is_training=training, reuse=True))

    def conv_layer(self, x, filter_size, num_filters, stride,
                   name,normalization_fn = tcl.batch_norm,activation_fn = tf.nn.relu,use_batch=True):
        with tf.variable_scope(name):
            # x = tf.expand_dims(x, 0)
            #num_in_channel = x.get_shape().as_list()[-1];
            #shape = [filter_size, filter_size, num_in_channel, num_filters];
            #W = self.weight_variable(shape, stddev=stddev);
            # tf.layers.dropout(W, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);

            #B = self.bias_variable(shape=[num_filters], stddev=stddev);
            # tf.layers.dropout(B, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);

            #tf.summary.histogram("weight", W);

            layer = tcl.conv2d(x, kernel_size=filter_size,num_outputs=num_filters
                               ,stride=stride, padding="SAME",);
            layer = tf.layers.dropout(layer, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAININGDO);
            #layer = tf.layers.batch_normalization(layer);
            # if(use_batch==True):
            #     #layer=tcl.batch_norm(layer,center=True,decay=0.5,scale=True,is_training=Config.IS_TRAINING
            #      #                    );
            #     layer = tf.nn.batch_normalization(x=layer,mean=0.0,variance=1.0,offset=True,scale=True,
            #                                       variance_epsilon=1e-8);

            layer = activation_fn(layer);
            #layer = self.Batch_Normalization(layer, training=Config.IS_TRAINING, name=name + "_Batch");

            #layer += B;
            return layer;

    # def conv_layer(self,input, kernel, filter, stride=1, name="_conv"):
    #     with tf.name_scope(name):
    #         network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,kernel_initializer=tf.random_normal_initializer(stddev=0.05),
    #                                    padding='SAME')
    #         return network

    def mlp_layer(self, x, num_units, name, use_relu=True, stddev=0.1):
        """
        Create a fully-connected layer
        :param x: input from previous layer
        :param num_units: number of hidden units in the fully-connected layer
        :param name: layer name
        :param use_relu: boolean to add ReLU non-linearity (or not)
        :return: The output array
        """
        with tf.variable_scope(name):
            in_dim = x.get_shape()[2]
            W = self.weight_variable(shape=[in_dim, num_units], stddev=stddev)
            # tf.layers.dropout(W, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            tf.summary.histogram('weight', W)
            #b = self.bias_variable(shape=[num_units], stddev=stddev)
            # tf.layers.dropout(b, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            #tf.summary.histogram('bias', b)
            layer = tf.matmul(x, W)
            #layer += b
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

    def fc_layer(self, x, num_units, name, use_relu=True, stddev=0.05):
        """
        Create a fully-connected layer
        :param x: input from previous layer
        :param num_units: number of hidden units in the fully-connected layer
        :param name: layer name
        :param use_relu: boolean to add ReLU non-linearity (or not)
        :return: The output array
        """
        with tf.variable_scope(name):
            in_dim = x.get_shape()[1]
            W = self.weight_variable(shape=[in_dim, num_units], stddev=stddev)
            tf.summary.histogram('weight', W)
           # b = self.bias_variable(shape=[num_units], stddev=stddev)
            # tf.summary.histogram('bias', b)
            layer = tf.matmul(x, W)
            #layer += b
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

    def flatten_layer(self, layer):
        """
        Flattens the output of the convolutional layer to be fed into fully-connected layer
        :param layer: input array
        :return: flattened array
        """
        with tf.variable_scope('Flatten_layer'):
            layer_shape = layer.get_shape()
            num_features = layer_shape[1:4].num_elements()
            layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    def bottleneckLayer(self,input,name,acitavtion_fn = tf.nn.relu):
        with tf.variable_scope(name):
            c = tcl.batch_norm(input);
            c = acitavtion_fn(c);
            c =  tcl.conv2d(c,kernel_size = 1,num_outputs= 4*self.growthRate,stride=1);
            c = tf.layers.dropout(c,rate = Config.DROPOUT_RATIO,training=Config.IS_TRAINING);

            c = tcl.batch_norm(c);
            c = acitavtion_fn(c);
            c = tcl.conv2d(c, kernel_size = 3,num_outputs= self.growthRate,stride=1);
            c = tf.layers.dropout(c, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            return  c;


    def transitionLayer(self,input,name,activation_fn = tf.nn.relu,half=True):
        with tf.variable_scope(name):
            c = tcl.batch_norm(input);
            c = activation_fn(c);
            inChannel = c.shape[-1];
            features = inChannel.value;
            if(features %2!=0):
                features -=1;
            if(half == True):
                c = tcl.conv2d(c, kernel_size = 1,num_outputs= 0.5*features,stride=1);
            c = self.max_pool(c,2,2,name="_avgPool");
            return c;


    def denseBlock(self,input,name,size):
        with tf.variable_scope(name):
            x =  self.bottleneckLayer(input,"_bottleneck" + str(0));
            x = tf.concat([input,x],axis = 3);
            for i in range(1,size):
                k = self.bottleneckLayer(x,"_bottleneck" + str(i));
                x = tf.concat([x,k],axis = 3);

            return x;

    def resBlock(self,x, num_outputs, kernel_size=4, stride=1,
                 activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                 scope=None):
        assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
        with tf.variable_scope(scope, 'resBlock'):
            shortcut = x
            if stride != 1 or x.get_shape()[3] != num_outputs:
                shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                      activation_fn=None, normalizer_fn=None, scope='shortcut')
            x = tcl.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding='SAME')
            x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding='SAME')
            x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME',
                           normalizer_fn=None)

            x += shortcut
            x = normalizer_fn(x)
            x = activation_fn(x)
        return x

    def extractFeatures(self, features,coord):
        #coord = self.placeholders['VertexLocationsNumpy'];
        coordShape = Config.NETWORK_SIZE * Config.NETWORK_SIZE;

        with tf.variable_scope(self.name + "_ExtractFeature"):
            finalFeatures = None;
            # Load feature
            for c in range(len(features)):
                currentFeatures = features[c];
                shape = currentFeatures.get_shape().as_list();
                batch = shape[0];

                coordstf33 = [np.floor(coord[0]*(shape[1]/128)),np.floor(coord[1]*(shape[1]/128))];

                coords33floorStacked = np.zeros((batch, coordShape, 3), dtype=np.float);

                coords33floor = np.floor(coordstf33);

                for j in range(batch):
                    for i in range(coordShape):
                        coords33floorStacked[j][i] = [j, coords33floor[0], coords33floor[1]];

                coordstf33floortf = tf.convert_to_tensor(value=coords33floorStacked, dtype=tf.float32);

                coordstf33floortf = tf.cast(coordstf33floortf, dtype=tf.int32);

                v133 = tf.gather_nd(currentFeatures, [coordstf33floortf], name="_gather_{}33".format(i + 1));

                finalTemp = tf.squeeze([v133]);
                if (c is 0):
                    finalFeatures = finalTemp;
                else:
                    finalFeatures = tf.concat([finalFeatures, finalTemp], axis=1);
            # ----------------------------------------------------------------------------

            return finalFeatures;
            # ---------------------------------------------------------

    def _build(self):
        self.pred = self.buildNetwork();
        gtPoints = self.placeholders['labels'];
        self.loss, self.accuracy ,self.thresh = EMD(self.pred, self.placeholders);
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtPoints, logits=self.pred), name='loss')
        #correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(gtPoints, 1), name='correct_pred')
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        #self.loss += tf.losses.get_regularization_loss();

    def buildNetwork(self):
        input = self.placeholders['imageInput']
        # x = tf.expand_dims(x, 0)
        stride = 1;
        with tf.variable_scope("ImageNetwork_"):

            # with arg_scope([tcl.batch_norm],is_training = Config.IS_TRAINING,scale=True):
            #     with arg_scope([tcl.conv2d,tcl.conv2d_transpose],activation_fn=tf.nn.relu,
            #                    normalizer_fn = tcl.batch_norm,
            #                    biases_initializer = None,
            #                    padding='SAME',
            #                    weights_regularizer = tcl.l2_regularizer(0.002),
            #                    weights_initializer = tf.initializers.random_normal(stddev=0.01)):
            #         size = 16
            #         # x: s x s x 3
            #         se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1,)  # 128 x 128 x 16
            #         se = self.resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 64 x 64 x 32
            #         se = self.resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 64 x 64 x 32
            #         se = self.resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 32 x 32 x 64
            #         se = self.resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 32 x 32 x 64
            #         se = self.resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 16 x 16 x 128
            #         se = self.resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 16 x 16 x 128
            #         se = self.resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 8 x 8 x 256
            #         se = self.resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 8 x 8 x 256
            #
            #         pd = tcl.conv2d_transpose(se, size * 16, , stride=1)  # 8 x 8 x 256
            #         pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
            #         pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 16 x 16 x 128
            #         pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
            #         pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 32 x 32 x 64
            #         pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
            #         pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
            #
            #         pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
            #         pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
            #         pd = tcl.conv2d_transpose(pd, size, 4, stride=1)
            #         pd = tcl.conv2d_transpose(pd, 1, 4, stride=1)

            # with arg_scope([tcl.batch_norm], is_training=Config.IS_TRAINING, scale=True,
            #                updates_collections=None,
            #                decay=0.9,
            #                center=True,
            #                zero_debias_moving_mean=True
            #                ):
            #     with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
            #                    normalizer_fn=tcl.batch_norm,
            #                    biases_initializer=None,
            #                    padding='SAME',
            #                    weights_regularizer=tcl.l2_regularizer(0.0),
            #                    weights_initializer=tf.initializers.random_normal(stddev=0.01)):
            #         x = tcl.conv2d(x,kernel_size=3,num_outputs=self.growthRate,stride=1);
            #         x = tcl.batch_norm(x);
            #         x = tcl.conv2d(x,kernel_size = 3,num_outputs = 2*self.growthRate,stride=2);
            #
            #         x1=self.denseBlock(x,name="dense1",size=4);
            #         x=self.transitionLayer(x1,name="transition1");
            #
            #         x2=self.denseBlock(x,name="dense2",size=5);
            #         x=self.transitionLayer(x2,name="transition2");
            #
            #         x3=self.denseBlock(x,name="dense3",size=4);
            #         x = self.transitionLayer(x3, name="transition3",half=False);
            #         pd= tcl.conv2d_transpose(x,64,kernel_size=4,stride=1);#128x128
            #         pd= tcl.conv2d_transpose(pd,64,kernel_size=4,stride=2);#128x128
            #         pd= tcl.conv2d_transpose(pd,32,kernel_size=4,stride=1);#64x64
            #         pd= tcl.conv2d_transpose(pd,32,kernel_size=4,stride=2);#64x64
            #         pd= tcl.conv2d_transpose(pd,16,kernel_size=4,stride=1);
            #         pd= tcl.conv2d_transpose(pd,16,kernel_size=4,stride=2);
            #         pd = tcl.conv2d_transpose(pd, 8, kernel_size=4, stride=1);
            #         pd = tcl.conv2d_transpose(pd, 8, kernel_size=4, stride=2);
            #         pd = tcl.conv2d_transpose(pd, 3, kernel_size=4, stride=1,activation_fn=None);

            #x=self.Batch_Normalization(x3, training=Config.IS_TRAINING, name='linear_batch')
            #x=tf.nn.relu(x);
            #x=global_avg_pool(x);
            #flatten=self.flatten_layer(x);

            #out = self.fc_layer(flatten,500,name="final",use_relu=False);
            with arg_scope([tcl.batch_norm], is_training=Config.IS_TRAINING, scale=True,
                           decay=0.9,
                           center=True,
                           trainable=True
                           ):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.002),
                               weights_initializer=tf.initializers.random_normal(stddev=0.01),trainable=True):
                    #First scale
                    conv1_1 = tcl.conv2d(input, kernel_size=9, num_outputs=64, stride=1,padding="SAME",normalizer_fn =None);
                    pool1_1 = self.max_pool(conv1_1, ksize=2, stride=2, name="pool1_1");
                    conv1_2 = self.conv_layer(pool1_1, 9, 128, stride=1, name="conv1_2");
                    pool1_2 = self.max_pool(conv1_2, ksize=2, stride=2, name="pool1_2");
                    conv1_3 = self.conv_layer(pool1_2, 9, 256, stride=1, name="conv1_3");

                    upsample1 = tcl.conv2d_transpose(conv1_3, 256, kernel_size=8, stride=2);
                    conv1_5  = self.conv_layer(upsample1, 1, 128, stride, "conv1_5");
                    #upsample1=pool2+upsample1;
                    # -------------------------------------------------------------------------
            with arg_scope([tcl.batch_norm], is_training=Config.IS_TRAINING, scale=True,
                           decay=0.9,
                           center=True,
                           trainable=True
                           ):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.002),
                               weights_initializer=tf.initializers.random_normal(stddev=0.01), trainable=True):
                    # Second scale network

                    conv2_1 = tcl.conv2d(input, kernel_size=5, num_outputs=64, stride=1, padding="SAME", normalizer_fn=None);
                    pool2_1 = self.max_pool(conv2_1, ksize=2, stride=2, name="pool2_1");
                    conv2_2 = self.conv_layer(pool2_1, 5, 128, stride, "conv2_2");

                    concat1 = tf.concat([conv1_5,conv2_2],axis=3,name="Concat1");

                    conv2_3 = self.conv_layer(concat1, 5, 256, stride, "conv2_3");

                    upsample2 = tcl.conv2d_transpose(conv2_3, 256, kernel_size=4, stride=2);
                    conv2_4 = self.conv_layer(upsample2, 1, 128, stride, "conv2_4");
                    # -------------------------------------------------------------------------

            with arg_scope([tcl.batch_norm], is_training=Config.IS_TRAINING, scale=True,
                           decay=0.9,
                           center=True,
                           trainable=True
                           ):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.002),
                               weights_initializer=tf.initializers.random_normal(stddev=0.01), trainable=True,
                               ):

                    #Third scale network
                    conv3_1 = tcl.conv2d(input, kernel_size=3, num_outputs=128, stride=1, padding="SAME", normalizer_fn=None);
                    concat2 = tf.concat([conv3_1, conv2_4], axis=3, name="Concat2");
                    conv3_2 = self.conv_layer(concat2, 3, 128, stride, "conv3_2");
                    conv3_3 = self.conv_layer(conv3_2, 1, 64, stride, "conv3_3");
                    # -------------------------------------------------------------------------

                    #Final prediction
                    pd = tcl.conv2d_transpose(conv3_3, 64, kernel_size=4, stride=1);
                    pd = tcl.conv2d_transpose(pd, 1, kernel_size=4, stride=1,activation_fn=tf.nn.sigmoid);
                    #---------------------------------------------------------------------------
                    #conv6_3 = self.conv_layer(conv6_2, 1, 16, stride, "conv6_3");
                    #pd = self.conv_layer(conv6_3, 1, 1, stride, "conv6_4");
                    #pd = tcl.conv2d_transpose(pd, 1, kernel_size=2, stride=1);


                    #pool5 = tf.image.resize_bilinear(conv5_3,[128,128]);
                    # -----------------------------------------------------

                    #Third scale network
                    # conv6_1 = self.conv_layer(x, 3, 128, stride=1, name="conv6_1");
                    # #conv6_2 = self.conv_layer(conv6_1, 3, 128, stride, "conv6_2");
                    # conv6_3 = self.conv_layer(conv6_1, 1, 64, stride, "conv6_3");
                    # 
                    # concat = tf.concat([pool5, conv6_3], axis=3, name="Concat2");
                    # 
                    # conv7_1 = self.conv_layer(concat, 3, 128, stride, "conv7_1");
                    # #conv7_2 = self.conv_layer(conv7_1, 3, 128, stride, "conv7_2");
                    # conv7_3 = self.conv_layer(conv7_1, 1, 64, stride, "conv7_3");
                    # pd = self.conv_layer(conv7_3, 1, 1, stride, "conv7_4");
                    #------------------------------------------------------

                    #pd = tf.image.resize_bilinear(conv5_4, [128, 128]);
                    #pd = tcl.conv2d_transpose(conv7_3, 64, kernel_size=4, stride=2);
                    #pd = tcl.conv2d_transpose(pd, 1, kernel_size=4, stride=1);
                    #pd = tcl.conv2d_transpose(pd, 3, kernel_size=4, stride=1);


                    # conv1_1 = self.conv_layer(x, 3, 32, stride, "conv1_1");
                    # # tf.layers.dropout(conv1_1,rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv1_2 = self.conv_layer(conv1_1, 3, 32, stride, "conv1_2");
                    # # conv1_1bn = tf.layers.batch_normalization(conv1_1);
                    # conv1_3 = self.conv_layer(conv1_2, 1, 16, stride, "conv1_3");
                    # # tf.layers.dropout(conv1_2, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # pool1 = self.max_pool(conv1_3, ksize=2, stride=2, name="pool1");
                    # #feature1 = self.avg_pool(conv1_2,ksize=200,stride=1,padding="VALID",name="feature1");
                    # 
                    # #
                    # conv2_1 = self.conv_layer(pool1, 3, 32, stride, "conv2_1");
                    # # tf.layers.dropout(conv2_1, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv2_2 = self.conv_layer(conv2_1, 3, 32, stride, "conv2_2");
                    # # conv2_1bn = tf.layers.batch_normalization(conv2_1);
                    # conv2_3 = self.conv_layer(conv2_2, 1, 16, stride, "conv2_3");
                    # # tf.layers.dropout(conv2_2, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # pool2 = self.max_pool(conv2_3, ksize=2, stride=2, name="pool2");
                    # #feature2 = self.avg_pool(conv2_2, ksize=100, stride=1, padding="VALID",name="feature2");
                    # 
                    # #
                    # conv3_1 = self.conv_layer(pool2, 3, 32, stride, "conv3_1");
                    # # tf.layers.dropout(conv3_1, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # # conv3_1bn = tf.layers.batch_normalization(conv3_1);
                    # conv3_2 = self.conv_layer(conv3_1, 3, 32, stride, "conv3_2");
                    # # tf.layers.dropout(conv3_2, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv3_3 = self.conv_layer(conv3_2, 1, 16, stride, "conv3_3");
                    # # tf.layers.dropout(conv3_3, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # pool3 = self.avg_pool(conv3_3, ksize=2, stride=2, name="pool3");
                    # #feature3 = self.avg_pool(conv3_3, ksize=50, stride=1, padding="VALID",name="feature3");
                    # 
                    # #
                    # conv4_1 = self.conv_layer(pool3, 3, 64, stride, "conv4_1");
                    # # tf.layers.dropout(conv4_1, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # # conv4_1bn = tf.layers.batch_normalization(conv4_1);
                    # conv4_2 = self.conv_layer(conv4_1, 3, 64, stride, "conv4_2");
                    # # tf.layers.dropout(conv4_2, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv4_3 = self.conv_layer(conv4_2, 1, 32, stride, "conv4_3");
                    # # tf.layers.dropout(conv4_3, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # pool4 = self.avg_pool(conv4_3, ksize=2, stride=2, name="pool4");
                    # #feature4 = self.avg_pool(conv4_3, ksize=25, stride=1, padding="VALID",name="feature4");
                    # # #pool4bn = tf.layers.batch_normalization(pool4);
                    # 
                    # conv5_1 = self.conv_layer(pool4, 3, 128, stride, "conv5_1");
                    # # tf.layers.dropout(conv5_1, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv5_2 = self.conv_layer(conv5_1, 3, 128, stride, "conv5_2");
                    # # tf.layers.dropout(conv5_2, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # conv5_3 = self.conv_layer(conv5_2, 1, 64, stride, "conv5_3");
                    # # tf.layers.dropout(conv5_3, rate=Config.DROPOUT_RATIO,training=self.placeholders["isTraining"]);
                    # #pool5 = self.avg_pool(conv5_3, ksize=2, stride=2, name="pool5",padding="SAME");
                    # 
                    # pd = tcl.conv2d_transpose(conv5_3, 64, kernel_size=4, stride=1 );  # 128x128
                    # pd = tcl.conv2d_transpose(pd, 64, kernel_size=4, stride=2);  # 128x128
                    # pd = tcl.conv2d_transpose(pd, 32, kernel_size=4, stride=1);  # 64x64
                    # pd = tcl.conv2d_transpose(pd, 32, kernel_size=4, stride=2);  # 64x64
                    # pd = tcl.conv2d_transpose(pd, 16, kernel_size=4, stride=1);
                    # pd = tcl.conv2d_transpose(pd, 16, kernel_size=4, stride=2);
                    # pd = tcl.conv2d_transpose(pd, 8, kernel_size=4, stride=1);
                    # pd = tcl.conv2d_transpose(pd, 8, kernel_size=4, stride=2);
                    # pd = tcl.conv2d_transpose(pd, 3, kernel_size=4, stride=1);

            #
            # flatten = self.flatten_layer(pool5);
            # flatten=self.flatten_layer(flatten);
            # fc1 = self.fc_layer(flatten, 128, "FC1", True);
            #fc1 = tf.layers.dropout(fc1, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            #fc2 = self.fc_layer(fc1, 96, "FC2", False);

            #vertexFeatures = tf.expand_dims(fc1,[1]);
            #vertexFeatures = tf.tile(vertexFeatures,[1,400,1]);

            # vertexFeatures0 = self.extractFeatures(
            #     features=[tf.squeeze(x2),tf.squeeze(x3)],coord=[64,64]);
            #
            # vertexFeatures1 = self.extractFeatures(
            #     features=[tf.squeeze(x2), tf.squeeze(x3)],coord=[64,63]);
            # vertexFeatures2 = self.extractFeatures(
            #     features=[tf.squeeze(x2), tf.squeeze(x3)], coord=[63, 64]);
            # vertexFeatures = tf.add(vertexFeatures0,vertexFeatures1,name="_FeatureAdder");
            # vertexFeatures = tf.add(vertexFeatures,vertexFeatures2,name="AADD");
            # #vertexFeatures= vertexFeatures0 + vertexFeatures1;
            # self.features = vertexFeatures;
            #flatten = self.flatten_layer(vertexFeatures);
            #vertLoc = self.placeholders['vertexLocations'];
            #vertexFeatures = tf.concat([vertexFeatures, vertLoc], axis=2);
            #self.debugFeatures = vertexFeatures;

            #Point-net segmentation network
            # mlp1 = self.fc_layer(flatten, 64, "_MLP1", True);
            # mlp1 = tf.layers.dropout(mlp1, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            # mlp2 = self.fc_layer(mlp1, 64, "_MLP2", True);
            # mlp2 = tf.layers.dropout(mlp2, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            # mlp3 = self.fc_layer(mlp2, 32, "_MLP3", True);
            # mlp3 = tf.layers.dropout(mlp3, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            # mlp4 = self.fc_layer(mlp3,32,"_MLP4",True);
            # mlp4 = tf.layers.dropout(mlp4, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);
            #mlp5 = self.fc_layer(flatten, 16, "_MLP5", True);
            #mlp5 = tf.layers.dropout(mlp5,rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);

            #mlp5 = self.fc_layer(vertexFeatures, 4, "_MLP5", True);
            #mlp6 = self.fc_layer(vertexFeatures, 10, "_MLP6", False);

            #tf.layers.dropout(mlp6, rate=Config.DROPOUT_RATIO, training=Config.IS_TRAINING);

            #mlp6 = tf.squeeze(mlp6);

            return pd;
