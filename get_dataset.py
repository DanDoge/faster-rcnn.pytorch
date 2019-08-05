import numpy as np
from collections import OrderedDict
import os
from PIL import Image
from scipy.io import loadmat
import random
objs = OrderedDict()
for label in os.listdir("./data/pascal3d+/Annotations"):
    for img_path in os.listdir("./data/pascal3d+/Annotations/" + label):
        img_path = img_path.split(".")[0]
        anno_file = loadmat("./data/pascal3d+/Annotations/" + label + "/" + img_path + ".mat")
        for obj in anno_file["record"]["objects"][0][0][0]:
            try:
                viewpoint = int(obj["viewpoint"]["azimuth"][0][0][0][0] // 15)
                bbox = obj["bbox"][0]
            except:
                break
            if not img_path in objs:
                objs[img_path] = np.empty(0)
            objs[img_path] = np.append(objs[img_path], {"label":label, "viewpoint":viewpoint, "bbox":bbox})
                #objs[label] = np.append(objs[label], {"img_path":img_path.split(".")[0], "viewpoint":viewpoint, "bbox":bbox})

import pickle
pickle.dump(objs, open("objs.pkl", "wb"))
