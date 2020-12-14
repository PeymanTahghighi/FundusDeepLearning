# ==================================================================================
# ==================================================================================
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display
import datetime
import Config
import numpy as np
import Utils


# ==================================================================================
# ==================================================================================

# ----------------------------------------------------------------------------------
class FundusNet():
    def __init__(self, numBatchTrain):
        # Define learning rate and optimizers
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Config.LEARNING_RATE,
            decay_steps=25 * numBatchTrain, decay_rate=0.9, staircase=True);
        self.generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate_schedule);
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate_schedule);
        # ---------------------------------------------------------------------------------------

        # Define generator and discriminator models
        self.generator = self.build_generator();
        self.discriminator = self.build_discriminator();
        # ---------------------------------------------------------------------------------------

        # Defin checkpoint and logging
        self.logdir = "logs/train";
        self.summary_writer = tf.summary.create_file_writer(
            self.logdir);
        tf.summary.trace_on(graph=True, profiler=True)
        self.trace(tf.zeros((Config.BATCH_SIZE, 128, 128, 3)))
        with self.summary_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="loggraph");

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              discriminator=self.discriminator,
                                              generator=self.generator);
        self.chekcpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=100)
        # ----------------------------------------------------------------------------------------

        self.hiddenLayerWeights=[5.0,1.0,5.0,5.0]

    @tf.function
    def trace(self,x):
        return self.generator(x)

    def save_checkpoint(self, step):
        path = self.chekcpoint_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(step), path))
        pass

    def load_checkpoint(self):
        if self.chekcpoint_manager.latest_checkpoint:
            idx = self.chekcpoint_manager.latest_checkpoint.find('-');
            epoch = (int(self.chekcpoint_manager.latest_checkpoint[idx + 1:]));

            self.checkpoint.restore(self.chekcpoint_manager.latest_checkpoint)
            print('[INFO]Restored from epoch : {}'.format(int(epoch + 1)));
            return int(epoch + 1);
        return 0;

    def Unet(self,inputs,postfix):
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False,name="DS1" + postfix),  # (bs, 64, 64, 64)
            self.downsample(128, 4,name="DS2" + postfix),  # (bs, 32, 32, 128)
            self.downsample(256, 4,name="DS3" + postfix),  # (bs, 16, 16, 256)
            self.downsample(512, 4,name="DS4" + postfix),  # (bs, 8, 8, 512)
            self.downsample(512, 4,name="DS5" + postfix),  # (bs, 4, 4, 512)
            self.downsample(512, 4,name="DS6" + postfix),  # (bs, 2, 2, 512)
            self.downsample(512, 4,name="DS7" + postfix),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True,name="US1" + postfix),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True,name="US2" + postfix),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True,name="U23" + postfix),  # (bs, 8, 8, 1024)
            self.upsample(256, 4,name="US4" + postfix),  # (bs, 16, 16, 1024)
            self.upsample(128, 4,name="US5" + postfix),  # (bs, 32, 32, 512)
            self.upsample(64, 4,name="US6" + postfix),  # (bs, 64, 64, 256)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh',name="convT" + postfix)  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        return x;
        pass

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3]);
        out = self.Unet(inputs=inputs,postfix="1");
        out1 = self.Unet(inputs=out,postfix="2");
        out2 = self.Unet(inputs=out1,postfix="3");

        final = tf.keras.layers.Average()([out,out1,out2]);

        return tf.keras.Model(inputs=inputs, outputs=final)

        pass

    def build_discriminator(self):

        initializer = tf.random_normal_initializer(0., 0.02)

        '''
            Uncomment to use PatchGAN.
        '''
        # inp = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
        # tar = tf.keras.layers.Input(shape=[128, 128, 3], name='target_image')
        #
        # x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
        #
        #
        # down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        # down1Act = tf.keras.layers.LeakyReLU(0.2)(down1);
        # down2 = self.downsample(128, 4,apply_batchnorm=True)(down1Act)  # (bs, 64, 64, 128)
        # down2Act = tf.keras.layers.LeakyReLU(0.2)(down2);
        # down3 = self.downsample(256, 4,apply_batchnorm=True)(down2Act)  # (bs, 32, 32, 256)
        # down3Act = tf.keras.layers.LeakyReLU(0.2)(down3)
        #
        # zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3Act)  # (bs, 34, 34, 256)
        # conv = tf.keras.layers.Conv2D(512, 4, strides=1,
        #                               kernel_initializer=initializer,
        #                               use_bias=False,
        #                               kernel_regularizer=tf.keras.regularizers.l2(0.0)
        #                               )(zero_pad1)  # (bs, 31, 31, 512)
        #
        # batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        #
        # leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm1)
        #
        # zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        #
        # last = tf.keras.layers.Conv2D(1, 4, strides=1,
        #                               kernel_initializer=initializer,
        #                               kernel_regularizer=tf.keras.regularizers.l2(0.0),
        #                               activation = tf.keras.layers.Activation('sigmoid')
        #                               )(zero_pad2)  # (bs, 30, 30, 1)
        #
        # return tf.keras.Model(inputs=[inp, tar], outputs=[last,down1,down2,down3,conv])


        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
        conv1 = self.conv2d(filters=64, kernel_size=3, stride=1, initializer=initializer,
                            apply_batchnorm=False)(inp);
        # out
        conv2 = self.conv2d(filters=128, kernel_size=3, stride=2, initializer=initializer)(conv1);
        conv3 = self.conv2d(filters=128, kernel_size=3, stride=1, initializer=initializer)(conv2);
        conv4 = self.conv2d(filters=256, kernel_size=3, stride=2, initializer=initializer)(conv3);
        # out

        conv5 = self.conv2d(filters=256, kernel_size=3, stride=1, initializer=initializer)(conv4);
        conv6 = self.conv2d(filters=512, kernel_size=3, stride=2, initializer=initializer)(conv5);
        # out

        conv7 = self.conv2d(filters=512, kernel_size=3, stride=1, initializer=initializer)(conv6);
        conv8 = self.conv2d(filters=512, kernel_size=3, stride=2, initializer=initializer)(conv7);
        # out

        conv9 = self.conv2d(filters=8, kernel_size=3, stride=2, initializer=initializer)(conv8);

        flatten = tf.keras.layers.Flatten()(conv9);
        out = self.fullyConnected(1, initializer=initializer, use_activation=True,
                                  activation_func=tf.keras.layers.Activation('sigmoid') ,apply_dropout=False)(flatten);

        return tf.keras.Model(inputs=inp, outputs=[out,conv1,conv4,conv6,conv8]);


    def downsample(self, filters, size, name, apply_batchnorm=True, trainable = True):
        with tf.name_scope(name) as scope:
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential(name=name)
            result.add(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                                    trainable = trainable,name="Conv2D"))

            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization(trainable = trainable))

            result.add(tf.keras.layers.Dropout(Config.DROPOUT_RATIO))

            return result

    def upsample(self, filters, size, name, apply_dropout=False, stride=2, activation_func=tf.keras.layers.LeakyReLU(0.2),trainable = True):
        with tf.name_scope(name) as scope:
            initializer = tf.random_normal_initializer(0., 0.02)

            result = tf.keras.Sequential(name=name)
            result.add(
                tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                use_bias=False,
                                                kernel_regularizer=tf.keras.regularizers.l2(0.0),
                                                trainable = trainable,
                                                name="ConvTranspose2D"
                                                ))

            result.add(tf.keras.layers.BatchNormalization(trainable = trainable))

            if apply_dropout and trainable is True:
                result.add(tf.keras.layers.Dropout(Config.DROPOUT_RATIO))

            result.add(activation_func)

            return result

    def conv2d(self, filters, kernel_size, stride, initializer, apply_batchnorm=False
               , activation_func=tf.keras.layers.LeakyReLU(0.2), trainable = True):
        
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                                          kernel_initializer=initializer, use_bias=False, trainable = trainable));
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization(trainable = trainable))

        result.add(activation_func)

        return result

    def maxPool2D(self, pool_size, stride=2):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride, padding='same');

    def fullyConnected(self, out, initializer, activation_func=tf.keras.layers.Activation('relu'),
                       use_bias=False, use_activation=True, apply_dropout=True):
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.Dense(units=out,
                                         kernel_initializer=initializer, use_bias=use_bias));

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(Config.DROPOUT_RATIO))

        if (use_activation):
            result.add(activation_func)

        return result

    @tf.function
    def generator_loss(self, disc_generated_output, gen_output, target, batch_size, gen_hidden_layers,target_hidden_layers):

        #gan loss
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        gan_loss = 0.5*tf.reduce_mean(tf.square(disc_generated_output-1));

        # mean absolute error
        epsilon = 0.0001;
        
        l2_loss = tf.reduce_mean(tf.pow((target - gen_output),2.0));

        #preceptual loss
        loss1 = tf.reduce_mean(tf.abs(gen_hidden_layers[0] - target_hidden_layers[0])) * self.hiddenLayerWeights[0];
        loss2 = tf.reduce_mean(tf.abs(gen_hidden_layers[1] - target_hidden_layers[1])) * self.hiddenLayerWeights[1];
        loss3 = tf.reduce_mean(tf.abs(gen_hidden_layers[2] - target_hidden_layers[2])) * self.hiddenLayerWeights[2];
        loss4 = tf.reduce_mean(tf.abs(gen_hidden_layers[3] - target_hidden_layers[3])) * self.hiddenLayerWeights[3];
        preceptualLoss = (loss1 + loss2 + loss3 + loss4);

        ssim = tf.reduce_mean(tf.image.ssim(gen_output,target,max_val = 1.0));

        psnr= Utils.PSNR(l2_loss);


        loss = 100.0*(l2_loss) + gan_loss + 50.0*preceptualLoss;

        return loss, l2_loss, gan_loss, ssim, preceptualLoss, psnr

    @tf.function
    def discriminator_loss(self, disc_real_output, disc_generated_output,gen_hidden_layers,target_hidden_layers):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        #perceptual loss
        loss1 = tf.reduce_mean(tf.abs(gen_hidden_layers[0] - target_hidden_layers[0])) * self.hiddenLayerWeights[0];
        loss2 = tf.reduce_mean(tf.abs(gen_hidden_layers[1] - target_hidden_layers[1])) * self.hiddenLayerWeights[1];
        loss3 = tf.reduce_mean(tf.abs(gen_hidden_layers[2] - target_hidden_layers[2])) * self.hiddenLayerWeights[2];
        loss4 = tf.reduce_mean(tf.abs(gen_hidden_layers[3] - target_hidden_layers[3])) * self.hiddenLayerWeights[3];
        preceptualLoss = (loss1 + loss2 + loss3 + loss4 );

        ganLoss = 0.5*tf.reduce_mean(tf.square(disc_generated_output)) + 0.5*tf.reduce_mean(tf.square(disc_real_output-1));
        return ganLoss + preceptualLoss;

    @tf.function
    def train_generator_step(self, input_image, target, globalStep, batch_size):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_generated_output,gen_f1,gen_f2,gen_f3,gen_f4 = self.discriminator(gen_output, training=False)
            _,target_f1,target_f2,target_f3,target_f4 = self.discriminator(target, training=False)

            loss, gen_l2_loss, gan_loss, ssim, perceptualLoss, psnr = self.generator_loss(disc_generated_output,
                                                                        gen_output, target, batch_size=batch_size,
                                                                        gen_hidden_layers=[gen_f1,gen_f2,gen_f3,gen_f4],
                                                                        target_hidden_layers=[target_f1,target_f2,target_f3,target_f4])

        generator_gradients = gen_tape.gradient(loss,
                                                self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gan_loss', gan_loss, step=globalStep)
            tf.summary.scalar('ssimTrain', ssim, step=globalStep)
            tf.summary.scalar('perceptualTrain', perceptualLoss, step=globalStep)
            tf.summary.scalar('psnrTrain', psnr, step=globalStep)
            tf.summary.scalar('l2loss_Train', gen_l2_loss, step=globalStep)
            tf.summary.scalar('loss_Train', loss, step=globalStep)
        return loss, ssim;

    @tf.function
    def train_discriminator_step(self, input_image, target, globalStep):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=False)

            disc_real_output,target_f1,target_f2,target_f3,target_f4 = self.discriminator(target, training=True)
            disc_generated_output,gen_f1,gen_f2,gen_f3,gen_f4 = self.discriminator(gen_output, training=True)



            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output,
                                                gen_hidden_layers=[gen_f1, gen_f2, gen_f3,gen_f4],
                                                target_hidden_layers=[target_f1, target_f2, target_f3,target_f4]
                                                )

        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('disc_loss', disc_loss, step=globalStep)
        return disc_loss;

    @tf.function
    def valid_step(self, input_image, target, globalStep, batch_size):

        gen_output = self.generator(input_image, training=False)

        disc_real_output, target_f1, target_f2, target_f3, target_f4 = self.discriminator(target, training=False)
        disc_generated_output, gen_f1, gen_f2, gen_f3, gen_f4 = self.discriminator(gen_output, training=False)


        loss, gen_l2_loss, gan_loss, ssim, perceptualLoss, psnr = self.generator_loss(disc_generated_output, gen_output,
                                                                    target, batch_size=batch_size,
                                                                    gen_hidden_layers=[gen_f1, gen_f2, gen_f3, gen_f4],
                                                                    target_hidden_layers=[target_f1, target_f2,
                                                                                          target_f3, target_f4]
                                                                    )
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output,
                                            gen_hidden_layers=[gen_f1, gen_f2, gen_f3, gen_f4],
                                            target_hidden_layers=[target_f1, target_f2,
                                                                  target_f3, target_f4]
                                            )

        with self.summary_writer.as_default():
            tf.summary.scalar('gan_loss_valid', gan_loss, step=globalStep)
            tf.summary.scalar('ssimValid', ssim, step=globalStep)
            tf.summary.scalar('perceptualValid', perceptualLoss, step=globalStep)
            tf.summary.scalar('psnrValid', psnr, step=globalStep)
            tf.summary.scalar('l2lossValid', gen_l2_loss, step=globalStep)
            tf.summary.scalar('lossValid', loss, step=globalStep)
            tf.summary.scalar('disc_loss_valid', disc_loss, step=globalStep)

        return loss, ssim;

    pass

    @tf.function
    def test_step(self, input_image):

        gen_output = self.generator(input_image, training=False)
        return gen_output;

    pass
# ----------------------------------------------------------------------------------
