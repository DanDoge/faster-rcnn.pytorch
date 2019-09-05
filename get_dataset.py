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
                    elevation = int(obj["viewpoint"]["elevation"][0][0][0][0] // 15)
                    bbox = obj["bbox"][0]
                except:
                    break
                if not img_path in objs:
                    objs[img_path] = np.empty(0)
                objs[img_path] = np.append(objs[img_path], {"label":label, "viewpoint":viewpoint, "elevation":elevation, "bbox":bbox})
                    #objs[label] = np.append(objs[label], {"img_path":img_path.split(".")[0], "viewpoint":viewpoint, "bbox":bbox})

    import pickle
    pickle.dump(objs, open("objs.pkl", "wb"), protocol=2)

def gen_objs_imagenet(split):
    objs = OrderedDict()
    for label in os.listdir("./data/pascal3d+1.1/Annotations"):
        if not label.endswith("imagenet"):
            continue
        with open("./data/pascal3d+1.1/Image_sets/" + label + "_" + split + ".txt") as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
        for img_path in image_ids:
            img_path = img_path.split(".")[0]
            anno_file = loadmat("./data/pascal3d+1.1/Annotations/" + label + "/" + img_path + ".mat")
            for obj in anno_file["record"]["objects"][0][0][0]:
                try:
                    viewpoint = int(obj["viewpoint"]["azimuth"][0][0][0][0] // 15)
                    elevation = int((obj["viewpoint"]["elevation"][0][0][0][0] + 90) // 15)
                    bbox = obj["bbox"][0]
                except:
                    continue
                if not img_path in objs:
                    objs[img_path] = np.empty(0)
                if split == "train":
                    objs[img_path] = np.append(objs[img_path], {"label":label, "viewpoint":viewpoint, "elevation":elevation, "bbox":bbox})
                elif split == "val":
                    objs[img_path] = np.append(objs[img_path], {"name":label, "viewpoint":viewpoint, "bbox":bbox})
                    #objs[label] = np.append(objs[label], {"img_path":img_path.split(".")[0], "viewpoint":viewpoint, "bbox":bbox})

    import pickle
    pickle.dump(objs, open("objs_imagenet_" + split + ".pkl", "wb"), protocol=2)

def gen_real_images():
    objs = OrderedDict()
    for label in os.listdir("./data/real_image/"):
        for img_path in os.listdir("./data/real_image/" + label):
            objs[img_path] = np.empty(0)
            objs[img_path] = np.append(objs[img_path], {"label":label, "viewpoint":0, "bbox":[1, 20, 1, 20]})

    import pickle
    pickle.dump(objs, open("objs_realimages.pkl", "wb"))

def gen_annots_imagenet():
    objs = OrderedDict()
    for label in os.listdir("./data/pascal3d+1.1/Annotations"):
        if not label.endswith("imagenet"):
            continue
        with open("./data/pascal3d+1.1/Image_sets/" + label + "_" + split + ".txt") as f:
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
        for img_path in image_ids:
            img_path = img_path.split(".")[0]
            anno_file = loadmat("./data/pascal3d+1.1/Annotations/" + label + "/" + img_path + ".mat")
            for obj in anno_file["record"]["objects"][0][0][0]:
                try:
                    viewpoint = int(obj["viewpoint"]["azimuth"][0][0][0][0] // 15)
                    bbox = obj["bbox"][0]
                except:
                    continue
                if not img_path in objs:
                    objs[img_path] = np.empty(0)
                objs[img_path] = np.append(objs[img_path], {"name":label, "viewpoint":viewpoint, "bbox":bbox})
                    #objs[label] = np.append(objs[label], {"img_path":img_path.split(".")[0], "viewpoint":viewpoint, "bbox":bbox})

    import pickle
    pickle.dump(objs, open("./data/pascal3d+1.1/annotations_cache/annots_pascal3d.pkl", "wb"), protocol=2)

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
    pickle.dump(objs, open("annots_pascal_imagenet_val.pkl", "wb"), protocol=2)

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
    gen_objs_imagenet("train")
