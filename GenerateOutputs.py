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
import numpy as np

data = Fetcher.DataFetcher(imagePath="Dataset/fundus", heightmapPath="Dataset/heightmap")
data.load();

batch_count = int(np.ceil(data.getTotalDataSize() / Config.BATCH_SIZE));

net = FundusNet(numBatchTrain=0);
startEpoch = net.load_checkpoint();

#Create necessary output folder
if not os.path.exists("Output"):
    os.makedirs("Output");

for b in tqdm(range(batch_count)):
    fundus, heightmap, size = data.fetchData(Config.BATCH_SIZE);
    out = net.generator(fundus, training=False);
    out = out.numpy() * 255;
    heightmap = heightmap * 255;
    fundus = fundus *255;
    for i in range(size):
        cv2.imwrite(filename='Output/' + str((b * Config.BATCH_SIZE) + i) + "_out.png", img=out[i]);
        cv2.imwrite(filename='Output/' + str((b * Config.BATCH_SIZE) + i) + "_GT.png", img=heightmap[i]);
        cv2.imwrite(filename='Output/' + str((b * Config.BATCH_SIZE) + i) + "_Fundus.png", img=fundus[i]);
