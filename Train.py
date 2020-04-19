from Models import *
import tensorflow as tf
import Config
import pickle
import Fetcher
import cv2
import math
from matplotlib import pyplot as pp

sess = tf.Session();
sess.run(tf.global_variables_initializer());
feed_dict = dict();


data = Fetcher.DataFetcher("Dataset\heightmap","Dataset\\fundus");
data.load();
numData = data.getTrainDataSize();
numIterationTrain = int(np.ceil(numData / Config.BATCH_SIZE));
numIterationValid = int(np.ceil(data.getValidDataSize()));

placeholders = {
    'imageInput' : tf.placeholder(tf.float32, shape=(None,Config.IMG_WIDTH, Config.IMG_HEIGHT, 3),name="inputImage"),
    'labels': tf.placeholder(dtype=tf.float32, shape=(None,Config.IMG_WIDTH, Config.IMG_HEIGHT, 1),name="Grountruth"),
    'vertexLocations': tf.placeholder(dtype=tf.float32, shape=(None, Config.NETWORK_SIZE * Config.NETWORK_SIZE,2),
                             name="vertexLocations"),
    'dataSize': tf.placeholder(dtype=tf.float32, shape=(),
                             name="dataSize"),
}

vertexLocations = np.zeros((Config.NETWORK_SIZE * Config.NETWORK_SIZE,2),dtype=np.float);
vertexLocations[0]=[63,63];
# for i in range(0,Config.NETWORK_SIZE ):
#     for j in range(0,Config.NETWORK_SIZE):
#          vertexLocations[i + j * 20] = [i,j]
# koft = tf.convert_to_tensor(vertexLocations,tf.float32);
# koft = tf.reshape(koft,[-1,10000,2]);
# placeholders['VertexLocationsTensor'] = koft;
placeholders['VertexLocationsNumpy'] = vertexLocations;



model = GCN(placeholders=placeholders);

init = tf.global_variables_initializer();
sess.run(init);

saver = tf.train.Saver();
Trainwriter = tf.summary.FileWriter(logdir="./graphs/General/Train",graph=sess.graph);
Validwriter = tf.summary.FileWriter(logdir="./graphs/General/Valid",graph=sess.graph);
DetailwriterTrain = tf.summary.FileWriter(logdir="./graphs/Detail/Train",graph=sess.graph);
DetailwriterValid = tf.summary.FileWriter(logdir="./graphs/Detail/Valid",graph=sess.graph);

lossPerEpoch = [];
accPerEpoch = [];
saveInterval = 2;
global_step = 0;
useCheckpoint = True;
loaded = False;

file = open("Experiments/deafultArch_" + str(Config.LEARNING_RATE) + "_" + str(Config.BATCH_SIZE)+"_Tiny.txt","w+");



if useCheckpoint is True:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('tf_ckpts/'))
    if ckpt and ckpt.model_checkpoint_path:
        loaded=True;
        saver.restore(sess, ckpt.model_checkpoint_path)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    #print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    #print(variable_parameters)
    total_parameters += variable_parameters
print("Total parameters : {}".format(total_parameters));

if loaded is True:
    global_step = sess.run(model.global_step);

#Find epochs
global_step/=numIterationTrain;


#Define loss and accuracy per epoch
all_loss = np.zeros(numIterationTrain,dtype='float32');
all_acc = np.zeros(numIterationTrain,dtype='float32');

all_lossValid = np.zeros(numIterationValid, dtype='float32');
all_accValid = np.zeros(numIterationValid, dtype='float32');

allLossTensor = tf.placeholder(tf.float32, shape=(None),name="AllLoss");
epochLoss = tf.reduce_mean(allLossTensor);

allAccTensor = tf.placeholder(tf.float32, shape=(None),name="AllAcc");
epochAcc = tf.reduce_mean(allAccTensor);
#-----------------------------------------------------------------

#Define summary variables
lossSummary = tf.summary.scalar("Loss", epochLoss);
accSummary = tf.summary.scalar("Accuracy", epochAcc);
lossSummaryDetailValid = tf.summary.scalar("LossV", epochLoss);
accSummaryDetailValid = tf.summary.scalar("AccuracyV", epochAcc);
#------------------------------------------------------------------

print("[INFO]Beginning from epoch : {}".format(int(global_step)));
for epoch in range(int(global_step),Config.EPOCHS):
    print("[INFO]Begin epoch {}...".format(epoch));
    all_loss.fill(0);
    all_acc.fill(0);

    for iter in range(numIterationTrain):
        img, y_train,size = data.fetchTrain(batchSize=Config.BATCH_SIZE);
        #vertexLocationsTmp = np.tile(vertexLocations, (Config.BATCH_SIZE, 1, 1));

        feed_dict.update({placeholders['imageInput']: img})
        feed_dict.update({placeholders['labels']: y_train})
        feed_dict.update({Config.IS_TRAINING : True})
        feed_dict.update({Config.IS_TRAININGDO : True})
        #feed_dict.update({placeholders['vertexLocations'] : vertexLocationsTmp})
        feed_dict.update({placeholders['dataSize'] : size})

        _,loss,accuracy,pred,thresh,step,summaryOp,lr = sess.run([model.opt_op,model.loss,
                       model.accuracy,model.pred,model.thresh,model.global_step,model.summary_op,model.optimizer._lr],feed_dict=feed_dict);

        #cv2.imwrite("./visualT/test" + str(global_step + iter+epoch) + ".png", pred[0] * 255);
        #cv2.imwrite("./visualT/img" + str(global_step + iter+epoch) + ".png", img[0] * 255);
        #cv2.imwrite("./visualT/h" + str(global_step + iter+epoch) + ".png", y_train[0] * 255);
        acc = np.count_nonzero(thresh);
        all_loss[iter] = loss;
        all_acc[iter] = accuracy;
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        mean_acc = np.mean(all_acc[np.where(all_acc)])
        DetailwriterTrain.add_summary(summaryOp, step);

    if global_step  % saveInterval == 0:
        saver.save(sess, "tf_ckpts/", global_step=model.global_step);
        print("Saved checkpoint for step {}".format(int(global_step)))

    #Print loss and accuracy of this epoch
    epochLoss = np.mean(all_loss);
    epochAcc = np.mean(all_acc);
    print('Epoch loss = {} \t Epoch accuracy : {} \t LR: {} '.format(epochLoss, epochAcc,lr));


    #--------------------------------------------------------

    #Write loss and accuracy for this epoch.
    summaryT = sess.run(lossSummary,feed_dict={allLossTensor : all_loss});
    summaryA = sess.run(accSummary,feed_dict={allAccTensor : all_acc});

    Trainwriter.add_summary(summaryT, global_step);
    Trainwriter.add_summary(summaryA, global_step);
    #---------------------------------------------------------


    #vertexLocationsTmp = np.tile(vertexLocations, (data.getValidDataSize(), 1, 1));
    accuracyValid = 0;
    for i in range(numIterationValid):

        # Validation data
        img, y_train,size = data.fetchValid(batchSize=1);
        feed_dict.update({placeholders['imageInput']: img})
        feed_dict.update({placeholders['labels']: y_train})
        feed_dict.update({Config.IS_TRAINING : False})
        feed_dict.update({Config.IS_TRAININGDO : False})
        #feed_dict.update({placeholders['vertexLocations']: vertexLocationsTmp})
        feed_dict.update({placeholders['dataSize'] : size})

        lossValid,accValid, pred,summaryOp = sess.run(
        [model.loss,model.accuracy, model.pred,model.summary_op], feed_dict=feed_dict);

        if global_step  % saveInterval == 0:
            cv2.imwrite("./visual/test" + str(global_step + i) + ".png", pred[0] * 255);
            cv2.imwrite("./visual/img" + str(global_step + i) + ".png", img[0] * 255);
            cv2.imwrite("./visual/h" + str(global_step + i) + ".png", y_train[0] * 255);

        #curAcc = np.count_nonzero(thresh);
        #accuracyValid +=curAcc;
        #curAcc/=Config.BATCH_SIZE;

        summaryDV = sess.run(lossSummaryDetailValid, feed_dict={allLossTensor: lossValid});
        summaryDA = sess.run(accSummaryDetailValid, feed_dict={allAccTensor: accValid});
        DetailwriterValid.add_summary(summaryDV, (global_step*numIterationValid)+i);
        DetailwriterValid.add_summary(summaryDA,  (global_step*numIterationValid)+i);

        all_lossValid[i] = lossValid;
        all_accValid[i] = accValid;

    lossValid = np.mean(all_lossValid);
    accValid = np.mean(all_accValid);
    #accuracyValid = accuracyValid/(data.getValidDataSize()*Config.NETWORK_SIZE*Config.NETWORK_SIZE);
    # Write loss and accuracy for this epoch.
    summaryT = sess.run(lossSummary, feed_dict={allLossTensor: lossValid});
    summaryA = sess.run(accSummary, feed_dict={allAccTensor: accValid});

    Validwriter.add_summary(summaryT, global_step);
    Validwriter.add_summary(summaryA, global_step);
    # ---------------------------------------------------------

    # Print loss and accuracy of this epoch
    #lossValid = np.mean(all_lossValid);
    #accuracyValid = np.mean(all_accValid);
    print('Epoch valid loss = {} \t Epoch valid accuracy : {}'.format(lossValid, accValid));
    #-------------------------------------

    global_step += 1;

# -----------------------------------------------------

file.close();