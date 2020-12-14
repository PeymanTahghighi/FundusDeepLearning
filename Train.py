from Models import *
import tensorflow as tf
import Config
import pickle
import Fetcher
import cv2
import math
from matplotlib import pyplot as pp
import pydot
from tqdm.autonotebook import tqdm

tf.config.optimizer.set_jit(True)

data = Fetcher.DataFetcher(imagePath="Dataset/fundus", heightmapPath="Dataset/heightmap");
data.load();

batch_count_train = int(np.ceil(data.trainDataSize / Config.BATCH_SIZE));
batch_count_valid = int(np.ceil(data.validDataSize / Config.BATCH_SIZE));
batch_count_test = int(np.ceil(data.testDataSize / Config.BATCH_SIZE));

gen_to_disc_ratio = 3;

net = FundusNet(numBatchTrain=int(batch_count_train) * gen_to_disc_ratio);
net.generator.summary();
net.generator.load_weights(filepath="ckpt_weights1\\weights",by_name=True)

startEpoch = net.load_checkpoint();


# Define loss and accuracy per epoch
all_loss = np.zeros(batch_count_train, dtype='float32');
all_ssim = np.zeros(batch_count_train, dtype='float32');

all_lossValid = np.zeros(batch_count_valid, dtype='float32');
all_ssimValid = np.zeros(batch_count_valid, dtype='float32');

all_lossTest = np.zeros(batch_count_test, dtype='float32');
all_ssimTest = np.zeros(batch_count_test, dtype='float32');
# ----------------------------------------------------------------------------

test = False;
train = True;

if test == False:
    for epoch in range(startEpoch, Config.EPOCHS):
        if(train is True):
            # Train step
            print("\n===================================================================");
            print("Epoch: ", epoch)
            for g in tqdm(range(gen_to_disc_ratio)):
                print("\n[INFO]Train generator step {}...".format(g));
                print("Learning rate : ", net.generator_optimizer._decayed_lr(tf.float32).numpy());
                for i in tqdm(tf.range(batch_count_train)):
                    i = tf.cast(i, tf.int64);
                    fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                    size = tf.convert_to_tensor(size, tf.int64);
                    loss, ssim = net.train_generator_step(fundus, heightmap,
                                                                        globalStep=i + (epoch * batch_count_train),
                                                                        batch_size=size)

                    all_loss[i] = loss;
                    all_ssim[i] = ssim;

                epochLoss = np.mean(all_loss);
                epochSsim = np.mean(all_ssim);
                print("[TRAIN]epoch : {} \t loss : {} \t ssim : {}".format(epoch, epochLoss, epochSsim));

            # -------------------------------------------------------------------------------

            print("\n[INFO]Train discriminator step...");
            print("Learning rate : ", net.generator_optimizer._decayed_lr(tf.float32).numpy());
            for i in tqdm(tf.range(batch_count_train)):
                i = tf.cast(i, tf.int64);
                fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                size = tf.convert_to_tensor(size, tf.int64);
                gen = net.train_discriminator_step(fundus, heightmap,
                                                                     globalStep=i + (epoch * batch_count_train))
            # Validation step
            print("\n[INFO]Start validation step...")
            for i in tqdm(tf.range(batch_count_valid)):
                i = tf.cast(i, tf.int64);
                fundus, heightmap, size = data.fetchValid(Config.BATCH_SIZE);
                size = tf.convert_to_tensor(size, tf.int64);
                loss, accuracy = net.valid_step(fundus, heightmap,
                                                globalStep=i + (epoch * batch_count_valid), batch_size=size)

                all_lossValid[i] = loss;
                all_ssimValid[i] = accuracy;

            print()

            epochLossValid = np.mean(all_lossValid);
            epochSSIMValid = np.mean(all_ssimValid);
            print("[VALID]epoch : {} \t loss : {} \t ssim : {}".format(epoch, epochLossValid, epochSSIMValid));
            # -------------------------------------------------------------------------------

           
            print("\n===================================================================");

        # Debug every 5 epoch
        if (epoch % 5 == 0):
            print("\n[INFO]Start debugging step...");
            for b in range(batch_count_train):
                fundus, heightmap, size = data.fetchTrain(Config.BATCH_SIZE);
                pred = net.generator(fundus, training=False);
                prednp = pred.numpy() * 255;
                heightmap = heightmap  * 255;
                fundus = fundus  * 255;
                a = len(prednp);
                for i in range(size):
                    cv2.imwrite(filename='visualT/' + str((b * batch_count_train) + i) + ".png", img=prednp[i]);
                    cv2.imwrite(filename='visualT/' + str((b * batch_count_train) + i) + "_GT.png", img=heightmap[i]);

            for b in range(batch_count_valid):
                fundus, heightmap, size = data.fetchValid(Config.BATCH_SIZE);
                pred = net.generator(fundus, training=False);
                prednp = pred.numpy() * 255;
                heightmap = heightmap  * 255;
                fundus = fundus  *255;
                a = len(prednp);
                for i in range(size):
                    cv2.imwrite(filename='visual/' + str((b * batch_count_valid) + i) + ".png", img=prednp[i]);
                    cv2.imwrite(filename='visual/' + str((b * batch_count_valid) + i) + "_GT.png", img=heightmap[i]);
                       
            print("\n[INFO]Finished debugging step...");
        # ------------------------------------------------------------------------------
        net.save_checkpoint(step=epoch);
        net.generator.save_weights(filepath="ckpt_weights2\\weights",save_format="h5");
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
        heightmap = heightmap  * 255;
        fundus = fundus  * 255;

        cv2.imwrite(filename='visualTest/' + str(b+50) + "_Predicted.png", img=prednp[0]);
        cv2.imwrite(filename='visualTest/' + str(b+50) + "_GT.png", img=heightmap[0]);
        cv2.imwrite(filename='visualTest/' + str(b+50) + "_Fundus.png", img=fundus[0]);

    print()
    print("\n[INFO]End test step");
