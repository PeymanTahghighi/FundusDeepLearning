import keras
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import os
import Config
from imutils import paths
import random
from keras.models import Model
import pickle

# Load VGG-16 Model
print("[INFO]Loading VGG-16 network...");
model = VGG16(weights='imagenet', include_top=False);

block3_conv3 = Model(inputs=model.input, outputs=model.get_layer('block3_conv3').output);
block4_conv3 = Model(inputs=model.input, outputs=model.get_layer('block4_conv3').output);
block5_conv3 = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output);
# ---------------------------------------------------------------------------

# Given X and Y coordinates returns the 4 corner points.
def getCorners(width, height, X, Y):
    tmpX = (width / Config.MAX_WIDTH) * X;
    tmpY = (height / Config.MAX_HEIGHT) * Y;

    topLeftX = np.floor(tmpX);
    topLeftY = np.floor(tmpY);

    topRightX = np.ceil(tmpX);
    topRightY = topLeftY;

    bottomLeftX = topLeftX;
    bottomLeftY = np.ceil(tmpY);

    bottomRightX = topRightX;
    bottomRightY = bottomLeftY;

    return (topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY);
# ---------------------------------------------------------------------------

# Extract features from images and store them.
def extractFeaturesAndStore():
    for split in (Config.TRAIN, Config.TEST, Config.VAL):
        print("[INFO]Processing for {} ...".format(split));
        p = os.path.sep.join([Config.BASE_PATH, split]);
        imagePaths = list(paths.list_images(p));

        for i in range(0, len(imagePaths)):
            fileName = os.path.basename(imagePaths[i]);
            fileName = os.path.splitext(fileName);
            fileName = fileName[0];

            # Load and process image.
            image = load_img(imagePaths[i], target_size=(224, 224));
            image = img_to_array(image);
            image = np.expand_dims(image, axis=0);
            image = imagenet_utils.preprocess_input(image);
            # ---------------------------------------------------------

            # Extract features from block_3 conv_3
            features1 = block3_conv3.predict(image);
            storeDir = os.path.sep.join([Config.BASE_FEATURE_OUTPUT, split, fileName + "_B3C3"]);
            pickle.dump(features1, open(storeDir, "wb+"));
            # features1 = features1.reshape((features1.shape[0], 56 * 56 * 256));
            # --------------------------------------------------------------------------

            # Extract features from block_3 conv_3
            features2 = block4_conv3.predict(image);
            storeDir = os.path.sep.join([Config.BASE_FEATURE_OUTPUT, split, fileName + "_B4C3"]);
            pickle.dump(features2, open(storeDir, "wb+"));
            # features2 = features2.reshape((features2.shape[0], 28* 28*512));
            # -------------------------------------------------------------------------

            # Extract features from block_3 conv_3
            features3 = block5_conv3.predict(image);
            storeDir = os.path.sep.join([Config.BASE_FEATURE_OUTPUT, split, fileName + "_B5C3"]);
            pickle.dump(features3, open(storeDir, "wb+"));
            # features3 = features3.reshape((features3.shape[0], 14*14*512));
            # ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Given X and Y coordinates extract a feature vector of size 1280 and return.
def extractFeature(imgName, X, Y, mode="Training"):
    assert X < Config.MAX_WIDTH, "Width is above maximum";
    assert Y < Config.MAX_HEIGHT, "Height is above maximum";

    # Load feature files
    filePath = os.path.sep.join([Config.BASE_FEATURE_OUTPUT, mode, imgName]);
    features1 = pickle.load(open(filePath + "_B3C3", "rb"));
    features2 = pickle.load(open(filePath + "_B4C3", "rb"));
    features3 = pickle.load(open(filePath + "_B5C3", "rb"));

    features1 = np.array(features1, dtype="float");
    features2 = np.array(features2, dtype="float");
    features3 = np.array(features3, dtype="float");

    # Extract features from block_3 conv_3
    (topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY) = getCorners(
        features1.shape[1], features1.shape[2], X, Y);
    feature3_3_1 = features1[0][int(topLeftX)][int(topLeftY)];
    feature3_3_2 = features1[0][int(topRightX)][int(topRightY)];
    feature3_3_3 = features1[0][int(bottomRightX)][int(bottomRightY)];
    feature3_3_4 = features1[0][int(bottomLeftX)][int(bottomLeftY)];


    avgFeatures1 = np.zeros(shape=(features1.shape[3],),dtype="float");

    for i in range(0,features1.shape[3]):
        avgFeatures1[i] = (feature3_3_1[i] + feature3_3_2[i] + feature3_3_3[i] + feature3_3_4[i])/4.0;
    #--------------------------------------------

    # Extract features from block_4 conv_3
    (topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY) = getCorners(
        features2.shape[1], features2.shape[2], X, Y);
    feature4_3_1 = features2[0][int(topLeftX)][int(topLeftY)];
    feature4_3_2 = features2[0][int(topRightX)][int(topRightY)];
    feature4_3_3 = features2[0][int(bottomRightX)][int(bottomRightY)];
    feature4_3_4 = features2[0][int(bottomLeftX)][int(bottomLeftY)];

    avgFeatures2 = np.zeros(shape=(features2.shape[3],), dtype="float");

    for i in range(0, features2.shape[3]):
        avgFeatures2[i] = (feature4_3_1[i] + feature4_3_2[i] + feature4_3_3[i] + feature4_3_4[i]) / 4.0;
    # --------------------------------------------

    # Extract features from block_5 conv_3
    (topLeftX, topLeftY, topRightX, topRightY, bottomRightX, bottomRightY, bottomLeftX, bottomLeftY) = getCorners(
        features3.shape[1], features3.shape[2], X, Y);
    feature5_3_1 = features3[0][int(topLeftX)][int(topLeftY)];
    feature5_3_2 = features3[0][int(topRightX)][int(topRightY)];
    feature5_3_3 = features3[0][int(bottomRightX)][int(bottomRightY)];
    feature5_3_4 = features3[0][int(bottomLeftX)][int(bottomLeftY)];

    avgFeatures3 = np.zeros(shape=(features3.shape[3],), dtype="float");

    for i in range(0, features3.shape[3]):
        avgFeatures3[i] = (feature5_3_1[i] + feature5_3_2[i] + feature5_3_3[i] + feature5_3_4[i]) / 4.0;
    # --------------------------------------------

    #append vectors and return
    tmpVec = np.concatenate((avgFeatures1,avgFeatures2));
    finalFeaturesVec = np.concatenate((tmpVec,avgFeatures3));
    #---------------------------------------------

    return finalFeaturesVec;
    # ---------------------------------------------------------
# ----------------------------------------------------------------------------

extractFeature(imgName="0", X=10, Y=10, mode="Training");
