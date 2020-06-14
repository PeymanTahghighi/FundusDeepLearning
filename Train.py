from Models import *
import tensorflow as tf
import Config
import pickle
import Fetcher
import cv2
import math
from matplotlib import pyplot as pp
import pydot


def count(stop):
    i = 0
    while i < stop:
        yield i
        i += 1


# ds_counter = tf.data.Dataset.from_generator(count, args=[250], output_types=tf.int32, output_shapes = (), )
#
# a = ds_counter.take(50);
# a = a.batch(10);
# b = ds_counter.skip(100);
# b = b.batch(10);
#
# for count_batch in a:
#   print(count_batch.numpy())
#
# for count_batch in b:
#   print(count_batch.numpy())
#


# def load(image_file):
#     image = tf.io.read_file(image_file)
#     image = tf.image.decode_jpeg(image,channels=3)
#
#     image = tf.cast(image, tf.float32)
#     image = image / 255.0;
#
#     return image
#
#
# dataset_fundus = tf.data.Dataset.list_files('Dataset/fundus/*.jpg',);
# dataset_fundus = dataset_fundus.map(load,
#                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
# dataset_heightmap = tf.data.Dataset.list_files('Dataset/heightmap/*.jpg',);
# dataset_heightmap = dataset_heightmap.map(load,
#                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#
# dataset = tf.data.Dataset.zip((dataset_fundus,dataset_heightmap));
# dataset = dataset.shuffle(buffer_size = 1000,seed=1,reshuffle_each_iteration=False);
# #dataset = dataset.apply(tf.data.experimental.unique());
# data_count = tf.data.experimental.cardinality(dataset);
# data_count = data_count.numpy();
#
# validDataset = dataset.take(np.ceil(data_count * 0.05));
# testDataset = dataset.skip(np.ceil(data_count * 0.1)).take(np.ceil(data_count * 0.1));
# trainDataset = dataset.skip(np.ceil(data_count * 0.99));`
#
# trainDataset = trainDataset.batch(Config.BATCH_SIZE);
# validDataset = validDataset.batch(Config.BATCH_SIZE);
# testDataset = testDataset.batch(Config.BATCH_SIZE);
#
# batch_count_train = tf.data.experimental.cardinality(trainDataset);
# batch_count_valid = tf.data.experimental.cardinality(validDataset);
# batch_count_test = tf.data.experimental.cardinality(testDataset);

data = Fetcher.DataFetcher(imagePath="Dataset/fundus", heightmapPath="Dataset/heightmap");
data.load();

batch_count_train = int(np.ceil(data.trainDataSize / Config.BATCH_SIZE));
batch_count_valid = int(np.ceil(data.validDataSize / Config.BATCH_SIZE));
batch_count_test = int(np.ceil(data.testDataSize / Config.BATCH_SIZE));

gen_to_disc_ratio = 3;

net = FundusNet(numBatchTrain=int(batch_count_train) * gen_to_disc_ratio);

startEpoch = net.load_checkpoint();

# Define loss and accuracy per epoch
all_loss = np.zeros(batch_count_train, dtype='float32');
all_acc = np.zeros(batch_count_train, dtype='float32');

all_lossValid = np.zeros(batch_count_valid, dtype='float32');
all_accValid = np.zeros(batch_count_valid, dtype='float32');

all_lossTest = np.zeros(batch_count_test, dtype='float32');
all_accTest = np.zeros(batch_count_test, dtype='float32');
# ----------------------------------------------------------------------------

test = False;
train = True;


if test == False:
    for epoch in range(startEpoch, Config.EPOCHS):
        if(train is True):
            # Train step
            print("\n===================================================================");
            print("Epoch: ", epoch)
            for g in range(gen_to_disc_ratio):
                print("\n[INFO]Train generator step {}...".format(g));
                print("Learning rate : ", net.generator_optimizer._decayed_lr(tf.float32).numpy());
                for i in tf.range(batch_count_train):
                    i = tf.cast(i, tf.int64);
                    fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                    size = tf.convert_to_tensor(size, tf.int64);
                    loss, accuracy = net.train_generator_step(fundus, heightmap,
                                                                         globalStep=i + (epoch * batch_count_train),

                                                                         batch_size=size)

                    all_loss[i] = loss;
                    all_acc[i] = accuracy;

                epochLoss = np.mean(all_loss);
                epochAcc = np.mean(all_acc);
                print("[TRAIN]epoch : {} \t loss : {} \t accuracy : {}".format(epoch, epochLoss, epochAcc));

            # -------------------------------------------------------------------------------

            print("\n[INFO]Train discriminator step...");
            print("Learning rate : ", net.generator_optimizer._decayed_lr(tf.float32).numpy());
            for i in tf.range(batch_count_train):
                i = tf.cast(i, tf.int64);
                fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                size = tf.convert_to_tensor(size, tf.int64);
                gen = net.train_discriminator_step(fundus, heightmap,
                                                                     globalStep=i + (epoch * batch_count_train),
                                                                     batch_size=size)

                all_loss[i] = loss;
                all_acc[i] = accuracy;


            # Validation step
            print("\n[INFO]Start validation step...")
            for i in tf.range(batch_count_valid):
                i = tf.cast(i, tf.int64);
                fundus, heightmap, size = data.fetchValid(Config.BATCH_SIZE);
                size = tf.convert_to_tensor(size, tf.int64);
                loss, accuracy = net.valid_step(fundus, heightmap,
                                                globalStep=i + (epoch * batch_count_valid), batch_size=size)

                all_lossValid[i] = loss;
                all_accValid[i] = accuracy;

            print()

            epochLossValid = np.mean(all_lossValid);
            epochAccValid = np.mean(all_accValid);
            print("[VALID]epoch : {} \t loss : {} \t accuracy : {}".format(epoch, epochLossValid, epochAccValid));
            # -------------------------------------------------------------------------------

            net.save_checkpoint(step=epoch);
            print("\n===================================================================");

        print("\n[INFO]Start debugging step...")
        # Debug every 5 epoch
        if (epoch % 5 == 0):
            for b in range(batch_count_train):
                fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                pred = net.generator(fundus, training=False);
                prednp = pred.numpy() * 255;
                heightmap = heightmap * 255;
                fundus = fundus * 255;
                a = len(prednp);
                for i in range(size):
                    # prednp[i] = cv2.cvtColor(prednp[i], cv2.COLOR_RGB2BGR);
                    # heightmap[i] = cv2.cvtColor(heightmap[i], cv2.COLOR_RGB2BGR);
                    cv2.imwrite(filename='visualT/' + str((b * batch_count_train) + i) + ".png", img=prednp[i]);
                    cv2.imwrite(filename='visualT/' + str((b * batch_count_train) + i) + "_GT.png", img=heightmap[i]);
                    #cv2.imwrite(filename='visualT/' + str((b * batch_count_train) + i) + "_Fundus.png", img=fundus[i]);

            for b in range(batch_count_valid):
                fundus, heightmap, size = data.fetchValid(Config.BATCH_SIZE);
                pred = net.generator(fundus, training=False);
                prednp = pred.numpy() * 255;
                heightmap = heightmap * 255;
                fundus = fundus *255;
                a = len(prednp);
                for i in range(size):
                    # prednp[i] = cv2.cvtColor(prednp[i], cv2.COLOR_RGB2BGR);
                    # heightmap[i] = cv2.cvtColor(heightmap[i], cv2.COLOR_RGB2BGR);
                    cv2.imwrite(filename='visual/' + str((b * batch_count_valid) + i) + ".png", img=prednp[i]);
                    cv2.imwrite(filename='visual/' + str((b * batch_count_valid) + i) + "_GT.png", img=heightmap[i]);
                    #cv2.imwrite(filename='visual/' + str((b * batch_count_valid) + i) + "_Fundus.png", img=fundus[i]);
        # ------------------------------------------------------------------------------
        print("\n[INFO]Finished debugging step...")
else:
    print("[INFO]Start testing step")
    for b in range(50):
        fundus, heightmap = data.getRandomTraining();
        pred = net.generator(fundus, training=False);
        prednp = pred.numpy() * 255;
        heightmap = heightmap * 255;
        fundus = fundus * 255;

        cv2.imwrite(filename='visualTest/' + str(b) + "_Predicted.png", img=prednp[0]);
        cv2.imwrite(filename='visualTest/' + str(b ) + "_GT.png", img=heightmap[0]);
        cv2.imwrite(filename='visualTest/' + str(b) + "_Fundus.png", img=fundus[0]);

    for b in range(50):
        fundus, heightmap = data.getRandomValidation();
        pred = net.generator(fundus, training=False);
        prednp = pred.numpy() * 255;
        heightmap = heightmap * 255;
        fundus = fundus * 255;

        cv2.imwrite(filename='visualTest/' + str(b+50) + "_Predicted.png", img=prednp[0]);
        cv2.imwrite(filename='visualTest/' + str(b+50) + "_GT.png", img=heightmap[0]);
        cv2.imwrite(filename='visualTest/' + str(b+50) + "_Fundus.png", img=fundus[0]);


    print()
    print("\n[INFO]End test step");
