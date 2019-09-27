from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import Config
import numpy as np
import pickle
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

def loadDataSplit(splitPath):
    data = [];
    labels = [];

    for row in open(splitPath):
        row = row.strip().split(",");
        label = row[0];
        features = np.array(row[1:],dtype = "float");

        data.append(features);
        labels.append(label);

    data = np.array(data);
    labels = np.array(labels);

    return (data,labels);

def csvFeatureGenerator(inputPath,bs,numClasses,mode = "train"):
    f = open(inputPath,"r");
    while True:
        data = [];
        labels = [];
        while len(data) < bs:
            row = f.readline();
            if row == "":
                f.seek(0);
                row = f.readline();
                if mode == "eval":
                    break;
            row = row.strip().split(",");
            label = row[0];
            label = to_categorical(label,num_classes = numClasses);
            features = np.array(row[1:],dtype="float");

            data.append(features);
            labels.append(label);

        yield (np.array(data),np.array(labels));


le = pickle.loads(open(Config.LE_PATH,"rb").read());
trainingPath = os.path.sep.join([Config.BASE_CSV_FILE,"{}.csv".format(Config.TRAIN)]);
testingPath = os.path.sep.join([Config.BASE_CSV_FILE,"{}.csv".format(Config.TEST)]);
evalPath = os.path.sep.join([Config.BASE_CSV_FILE,"{}.csv".format(Config.VAL)]);

totalTrain = sum([1 for l in open(trainingPath)]);
totalVal = sum([1 for l in open(evalPath)]);
testLabels = [int(row.split(",")[0]) for row in open(testingPath)]
totalTest = len(testLabels)

print("[INFO]Generating train genrator...");
trainGen = csvFeatureGenerator(trainingPath,Config.BATCH_SIZE,len(Config.CLASSES),mode = "train");
print("[INFO]Generating test genrator...");
testGen = csvFeatureGenerator(testingPath,Config.BATCH_SIZE,len(Config.CLASSES),mode = "test");
print("[INFO]Generating evaluation genrator...");
evalGen = csvFeatureGenerator(evalPath,Config.BATCH_SIZE,len(Config.CLASSES),mode = "eval");

model = Sequential();
model.add(Dense(256,input_shape=(56*56*256,),activation="relu"));
model.add(Dense(16,activation="relu"));
model.add(Dense(len(Config.CLASSES),activation="softmax"));

opt = SGD(lr = 1e-3,momentum=0.9,decay=1e-3/25);
print("[INFO]Compiling model...");
model.compile(loss = "binary_crossentropy",optimizer=opt,metrics=["accuracy"]);
print("[INFO]Training network...");
H = model.fit_generator(trainGen,steps_per_epoch=totalTrain // Config.BATCH_SIZE,
                        validation_data=evalGen,
                        validation_steps=totalVal // Config.BATCH_SIZE,
                        epochs=25);
print("[INFO]Evaluation network...");
predIdxs = model.predict_generator(testGen,
	steps=(totalTest //Config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs,axis=1);
print(classification_report(testLabels,predIdxs,target_names=le.classes_));