import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import Config
import cv2
from keras.applications import imagenet_utils

class DataFetcher(object):
    def __init__(self, heightmapPath, imagePath):

        self.heightMapBasePath = heightmapPath;
        self.imageBasePath = imagePath;
        self.trainImage = [];
        self.trainHeight = [];
        self.testImage = [];
        self.testHeight = [];
        self.validImage = [];
        self.validHeight = [];
        self.loaded = False;
        self.trainIdx = 0;
        self.testIdx = 0;
        self.validIdx = 0;
        self.trainDataSize = 0;
        self.testDataSize = 0;
        self.validDataSize = 0;
        self.totalDataSize = 0;

    def load(self):
        # load all files

        imagefiles = os.listdir(self.imageBasePath);
        heightmapFiles = os.listdir(self.heightMapBasePath);
        numFiles = len(imagefiles);
        self.totalDataSize = numFiles;
        idx = np.arange(0, numFiles);
        np.random.seed(1);
        np.random.shuffle(idx);

        i = 0;
        self.trainDataSize = int(np.ceil(numFiles * 0.9));
        #self.trainDataSize = 100;
        #self.validDataSize= 50;
        #self.testDataSize =  int(np.ceil((numFiles - self.trainDataSize)));
        self.validDataSize = int(np.ceil((numFiles - self.trainDataSize) * 0.5));
        meanImage = load_img(path="meanImage.png");
        meanImage = img_to_array(meanImage);
        while i < self.trainDataSize:
            # Load image
            imgPath = os.path.sep.join([self.imageBasePath, imagefiles[idx[i]]]);
            img = cv2.imread(imgPath,cv2.IMREAD_COLOR);
            #img = img-meanImage;
            img = img / 255.0;
            self.trainImage.append(img);

            # ----------------------------------------------

            # Load heightmap
            heightmapPath = os.path.sep.join([self.heightMapBasePath, heightmapFiles[idx[i]]]);
            # heightmap = open(heightmapPath, "r", encoding="utf-8");
            # heightMapMat = np.zeros((Config.NETWORK_SIZE * Config.NETWORK_SIZE), np.float);
            # lines = heightmap.readlines();
            # for k in range(Config.NETWORK_SIZE * Config.NETWORK_SIZE):
            #     heightMapMat[k] = int(lines[k]);
            #     #heightMapMat[k] /= 255.0;
            #
            # #heightMapMat -= heightMapMat.min();
            img = cv2.imread(heightmapPath, cv2.IMREAD_COLOR);
            #img = np.expand_dims(img, axis=2);
            # img = img-meanImage;
            img = img / 255.0;

            self.trainHeight.append(img);
            # self.trainHeight.append(float(lines[0]));
            # ----------------------------------------------
            i += 1;
        #Load our validation data
        while i <= self.trainDataSize + self.validDataSize:
            # Load image
            imgPath = os.path.sep.join([self.imageBasePath, imagefiles[idx[i]]]);
            img = cv2.imread(imgPath,cv2.IMREAD_COLOR);
            #img = img - meanImage;
            img = img / 255.0;
            self.validImage.append(img);
            # ----------------------------------------------

            # Load heightmap
            heightmapPath = os.path.sep.join([self.heightMapBasePath, heightmapFiles[idx[i]]]);
            # heightmap = open(heightmapPath, "r", encoding="utf-8");
            # heightMapMat = np.zeros((Config.NETWORK_SIZE * Config.NETWORK_SIZE), np.float);
            # lines = heightmap.readlines();
            # for k in range(Config.NETWORK_SIZE * Config.NETWORK_SIZE):
            #     heightMapMat[k] = int(lines[k]);
            #     #heightMapMat[k] /= 255.0;

            img = cv2.imread(heightmapPath, cv2.IMREAD_COLOR);
            #img = np.expand_dims(img, axis=2);
            # img = img-meanImage;
            img = img / 255.0;

           # heightMapMat -= heightMapMat.min();
            self.validHeight.append(img);
            # self.testHeight.append(float(lines[0]));
            # ----------------------------------------------
            i += 1;

        # Load our Test data
        while i < 0:
            # Load image
            imgPath = os.path.sep.join([self.imageBasePath, imagefiles[idx[i]]]);
            img = load_img(path=imgPath);
            img = img_to_array(img);
           # img = img - meanImage;
            img = img / 255.0;
            self.testImage.append(img);
            # ----------------------------------------------

            # Load heightmap
            heightmapPath = os.path.sep.join([self.heightMapBasePath, heightmapFiles[idx[i]]]);
            # heightmap = open(heightmapPath, "r", encoding="utf-8");
            # heightMapMat = np.zeros((Config.NETWORK_SIZE * Config.NETWORK_SIZE), np.float);
            # lines = heightmap.readlines();
            # for k in range(Config.NETWORK_SIZE * Config.NETWORK_SIZE):
            #     heightMapMat[k] = float(lines[k]);
            #     #heightMapMat[k] /= 255.0;
            #
            # #heightMapMat -= heightMapMat.min();

            img = load_img(path=heightmapPath);
            img = img_to_array(img);
            # img = img-meanImage;
            img = img / 255.0;
            self.testHeight.append(img);
            # self.testHeight.append(float(lines[0]));
            # ----------------------------------------------
            i += 1;
        self.loaded = True;

    def fetchTrain(self, batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.trainDataSize - self.trainIdx < batchSize):
            size = self.trainDataSize - self.trainIdx;
        else:
            size = batchSize;
        retImg = np.zeros(shape=(size, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retheight = np.zeros(shape=(size, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(size):
            # tmp = np.zeros((Config.NETWORK_SIZE * Config.NETWORK_SIZE), dtype=np.float);
            #
            # tmp = self.trainHeight[i];
            # tmp = np.squeeze(tmp);
            retheight[i] = self.trainHeight[self.trainIdx];

            retImg[i] = self.trainImage[self.trainIdx];
            self.trainIdx += 1;

        if self.trainIdx == self.trainDataSize:
            self.trainIdx = 0;
        return retImg, retheight,size;


    def fetchTest(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        retImg = np.zeros(shape=(self.testDataSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 1), dtype=np.float32);
        retheight = np.zeros(shape=(self.testDataSize, Config.NETWORK_SIZE * Config.NETWORK_SIZE), dtype=np.float32);

        for i in range(self.testDataSize):
            # if self.testIdx == self.testDataSize:
            #     self.testIdx = 0;
            #     break;
            retheight[i] = self.testHeight[i];

            retImg[i] = self.testImage[i];
            #self.testIdx += 1;
        return retImg, retheight;

    def fetchValid(self, batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if (self.validDataSize - self.validIdx < batchSize):
            size = self.validDataSize - self.validIdx;
        else:
            size = batchSize;
        retImg = np.zeros(shape=(size, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retheight = np.zeros(shape=(size, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3
                                    ), dtype=np.float32);

        for i in range(size):
            retheight[i] = self.validHeight[self.validIdx];

            retImg[i] = self.validImage[self.validIdx];
            self.validIdx += 1;

        if self.validIdx == self.validDataSize:
            self.validIdx = 0;
        return retImg, retheight,size;

    def getTrainDataSize(self):
        return self.trainDataSize;


    def getTestDataSize(self):
        return self.testDataSize;

    def getTotalDataSize(self):
        return self.totalDataSize;


    def getRandomValidation(self):
        retImg = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retheight = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3
                                    ), dtype=np.float32);

        i = np.random.random_integers(low=0,high=self.validDataSize-1);
        retheight[0] = self.validHeight[i];
        retImg[0] = self.validImage[i];

        return retImg, retheight;

    def getRandomTraining(self):
        retImg = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retheight = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3
                                    ), dtype=np.float32);

        i = np.random.random_integers(low=0, high=self.trainDataSize - 1);
        retheight[0] = self.trainHeight[i];
        retImg[0] = self.trainImage[i];


        return retImg, retheight;

    def getValidDataSize(self):
        return self.validDataSize;
