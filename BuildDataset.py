import os
import Config
import shutil
from imutils import paths

for split in (Config.TRAIN , Config.TEST,Config.VAL):
    p = os.path.sep.join([Config.ORIG_INPUT_DATASET,split]);
    imagePaths = list(paths.list_images(p));

    for imagePath in imagePaths:

        sp = imagePath.split("_");
        fp = sp[0];
        fileName = sp[1];
        label = fp[len(fp)-1];
        dirPath = os.path.sep.join([Config.BASE_PATH,split,label]);
        if not os.path.exists(dirPath):
            os.makedirs(dirPath);

        p = os.path.sep.join([dirPath,fileName]);
        shutil.copy2(imagePath,p);