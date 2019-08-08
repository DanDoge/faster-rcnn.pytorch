import numpy as np
from collections import OrderedDict
import os
from PIL import Image
from scipy.io import loadmat
import random

def gen_objs():
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

def gen_annots():
    objs = {}
    for label in os.listdir("./data/pascal3d+/Annotations"):
        for img_path in os.listdir("./data/pascal3d+/Annotations/" + label):
            img_path = img_path.split(".")[0]
            anno_file = loadmat("./data/pascal3d+/Annotations/" + label + "/" + img_path + ".mat")
            for obj in anno_file["record"]["objects"][0][0][0]:
                try:
                    viewpoint = int(obj["viewpoint"]["azimuth"][0][0][0][0] // 15)
                    bbox = obj["bbox"][0]
                except:
                    continue
                if img_path not in objs:
                    objs[img_path] = []
                objs[img_path] = np.append(objs[img_path], {"name":label, "viewpoint":viewpoint, "bbox":bbox, "difficult":0})
    pickle.dump(objs, open("annots_pascal.pkl", "wb"))

def gen_imagesets():
    for label in os.listdir("./data/pascal3d+/Annotations"):
        with open("./data/pascal3d+/Imagesets/{}.txt".format(label), "w") as f:
            for img_path in os.listdir("./data/pascal3d+/Annotations/" + label):
                f.write(img_path.split(".")[0] + ' -1\n')

def gen_val():
    with open("./data/pascal3d+/ImageSets/val.txt", "w") as f:
        for label in os.listdir("./data/pascal3d+/Annotations"):
            for img_path in os.listdir("./data/pascal3d+/Annotations/" + label):
                f.write(img_path.split(".")[0] + '\n')

if __name__ == '__main__':
    gen_val()
