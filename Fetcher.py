import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import Config
import cv2
from keras.applications import imagenet_utils
from tqdm.autonotebook import tqdm
import pickle
import tensorflow as tf
import glob
from sklearn.utils import shuffle

class DataFetcher(object):
    def __init__(self, heightmapPath, imagePath):

        self.heightMapBasePath = heightmapPath;
        self.imageBasePath = imagePath;

        self.allFundus = [];
        self.allHeightmap=[];

        self.trainFundus = [];
        self.trainHeight = [];

        self.testFundus = [];
        self.testHeight = [];

        self.validFundus = [];
        self.validHeight = [];
        
        self.loaded = False;

        self.trainIdx = 0;
        self.testIdx = 0;
        self.validIdx = 0;
        self.dataIdx = 0;

        self.trainDataSize = 0;
        self.testDataSize = 0;
        self.validDataSize = 0;
        self.totalDataSize = 0;

    def load(self):

        imagefiles = os.listdir(self.imageBasePath);
        heightmapFiles = os.listdir(self.heightMapBasePath);
        numFiles = len(imagefiles);
        idx = np.arange(0, numFiles);
        np.random.seed(1);
        np.random.shuffle(idx);
        
        meanImage = load_img(path="meanImage.png");
        meanImage = img_to_array(meanImage);
        fileCounter = 0;

        for i in tqdm(range(numFiles)):

            # Load image
            imgPath = os.path.sep.join([self.imageBasePath, imagefiles[idx[i]]]);
            fundus = cv2.imread(imgPath,cv2.IMREAD_COLOR);
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8));
            lab = cv2.cvtColor(fundus, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            lab_planes[0] = clahe.apply(lab_planes[0])

            fundus = cv2.merge(lab_planes)

            fundus = cv2.cvtColor(fundus, cv2.COLOR_LAB2BGR)
            fundus = fundus / 255.0;

            # ----------------------------------------------

            # Load heightmap
            heightmapPath = os.path.sep.join([self.heightMapBasePath, heightmapFiles[idx[i]]]);
            heightmap = cv2.imread(heightmapPath, cv2.IMREAD_COLOR);
            heightmap = heightmap / 255.0;
            # ----------------------------------------------

            self.allFundus.append(fundus);
            self.allHeightmap.append(heightmap);
        
        augmented = [];
        for i in range(len(self.allFundus)):
            for j in range(-1,3,1):
                augmented.append([i,j]);

        self.totalDataSize = len(self.allFundus);

        augmented = shuffle(augmented,random_state=0);

        self.trainFundus = augmented[:int(np.ceil(len(augmented)*0.8))];
        self.trainHeight = augmented[:int(np.ceil(len(augmented)*0.8))];
        self.trainDataSize = len(self.trainFundus);

        self.validFundus = augmented[int(np.ceil(len(augmented)*0.8)):self.trainDataSize + int(np.ceil((len(augmented) - self.trainDataSize)*0.5))];
        self.validHeight = augmented[int(np.ceil(len(augmented)*0.8)):self.trainDataSize + int(np.ceil((len(augmented) - self.trainDataSize)*0.5))];
        self.validDataSize = len(self.validFundus);

        self.testFundus = augmented[self.trainDataSize + int(np.ceil((len(augmented) - self.trainDataSize)*0.5)):];
        self.testHeight = augmented[self.trainDataSize + int(np.ceil((len(augmented) - self.trainDataSize)*0.5)):];

        self.testDataSize = len(self.testFundus);
        
        self.loaded = True;

    def fetchTrain(self, batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.trainDataSize - self.trainIdx < batchSize):
            size = self.trainDataSize - self.trainIdx;
            replicate = batchSize - size;
        else:
            size = batchSize;

        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(size):
            f = self.allFundus[self.trainFundus[self.trainIdx][0]];
            code = self.trainFundus[self.trainIdx][1];
            h = self.allHeightmap[self.trainHeight[self.trainIdx][0]];
            if code == 2:
                retImg[i] = f;
                retHeight[i] = h;
            else:
                retImg[i] = cv2.flip(f,code);
                retHeight[i] = cv2.flip(h,code);
            
            self.trainIdx += 1;

        if self.trainIdx == self.trainDataSize:
            self.trainIdx = 0;
        return retImg, retHeight,size;


    def fetchTest(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.testDataSize - self.testIdx < batchSize):
            size = self.testDataSize - self.testIdx;
            replicate = Config.BATCH_SIZE - size;
        else:
            size = batchSize;

        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(size):
            f = self.allFundus[self.testFundus[self.testIdx][0]];
            code = self.testFundus[self.testIdx][1];
            h = self.allHeightmap[self.testHeight[self.testIdx][0]];
            if code == 2:
                retImg[i] = f;
                retHeight[i] = h;
            else:
                retImg[i] = cv2.flip(f,code);
                retHeight[i] = cv2.flip(h,code);
            
            self.testIdx += 1;

        if self.testIdx == self.testDataSize:
            self.testIdx = 0;
        return retImg, retHeight,size;

    def fetchRandomTest(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        
        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(batchSize):
            r = np.random.randint(0,self.testDataSize);
            f = self.allFundus[self.testFundus[r][0]];
            code = self.testFundus[r][1];
            h = self.allHeightmap[self.testHeight[r][0]];
            if code == 2:
                retImg[i] = f;
                retHeight[i] = h;
            else:
                retImg[i] = cv2.flip(f,code);
                retHeight[i] = cv2.flip(h,code);
        
        return retImg, retHeight;

    def fetchRandomValid(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        
        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(batchSize):
            r = np.random.randint(0,self.validDataSize);
            f = self.allFundus[self.validFundus[r][0]];
            code = self.validFundus[r][1];
            h = self.allHeightmap[self.validHeight[r][0]];
            if code == 2:
                retImg[i] = f;
                retHeight[i] = h;
            else:
                retImg[i] = cv2.flip(f,code);
                retHeight[i] = cv2.flip(h,code);
        
        return retImg, retHeight;

    def fetchValid(self, batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.validDataSize - self.validIdx < batchSize):
            size = self.validDataSize - self.validIdx;
            replicate = Config.BATCH_SIZE - size;
        else:
            size = batchSize;

        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(size):
            f = self.allFundus[self.validFundus[self.validIdx][0]];
            code = self.validFundus[self.validIdx][1];
            h = self.allHeightmap[self.validHeight[self.validIdx][0]];
            if code == 2:
                retImg[i] = f;
                retHeight[i] = h;
            else:
                retImg[i] = cv2.flip(f,code);
                retHeight[i] = cv2.flip(h,code);
            
            self.validIdx += 1;

        if self.validIdx == self.validDataSize:
            self.validIdx = 0;
        return retImg, retHeight,size;

    def getTrainDataSize(self):
        return self.trainDataSize;


    def getTestDataSize(self):
        return self.testDataSize;

    def getTotalDataSize(self):
        return len(self.allFundus);

    ''' 
        Fetch all data in sequential order
    '''
    def fetchData(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.totalDataSize - self.dataIdx < batchSize):
            size = self.totalDataSize - self.dataIdx;
            replicate = batchSize - size;
        else:
            size = batchSize;

        retImg = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retHeight = np.zeros(shape=(batchSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);

        for i in range(size):
            f = self.allFundus[self.dataIdx];
            h = self.allHeightmap[self.dataIdx];
            retImg[i] = f;
            retHeight[i] = h;
            
            self.dataIdx += 1;

        if self.dataIdx == len(self.allFundus):
            self.dataIdx = 0;
        return retImg, retHeight,size;


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
