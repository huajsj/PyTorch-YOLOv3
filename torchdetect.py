from __future__ import division

import numpy as np
from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image

class_path = "data/coco.names"
model_path="./checkpoints-0608/yolov3_tiny_all_99.pth"
img_path="./person.jpg"
image_size=320
image_channel=3

conf_thres=0.8
nms_thres=0.4

device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
model.eval()

classes = load_classes(class_path)
Tensor = torch.FloatTensor

img = Image.open(img_path).resize((image_size, image_size))
if 1:
    from torchvision import transforms
    '''
    my_preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor()])
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])])
    '''
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img, _ = pad_to_square(img, pad_value=0)# trainning dataset logic
    img = resize(img, image_size)
    img = img.unsqueeze(0)
    #img = np.expand_dims(img, 0)
    #img = torch.from_numpy(img.astype("float32"))
    img = Variable(img.type(Tensor))
    '''
    class trainset(Dataset):
        def __init__(self):
            a = 2

        def __getitem__(self, index):
            return "", img
        def __len__(self):
            return 1

    dataloader = DataLoader(trainset(), 
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    for i, (path, ig) in enumerate(dataloader):
        img = Variable(ig.type(Tensor))
        break
    print("--------------------")
    '''
    print(img)
else:
    #img = np.expand_dims(img, 0)
    #img = img.transpose((0, 3, 1, 2))
    #img = Variable(torch.from_numpy(img.astype("float32")))
    image_folder = "./data/samples2/"
    idata = ImageFolder(image_folder, img_size=image_size),
    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=image_size),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    for i, (path, ig) in enumerate(dataloader):
        img = Variable(ig.type(Tensor))
        break
    print(img)
'''
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            for l in range(img.shape[3]):
                if (img[i][j][k][l] != 0):
                    print(img[i][j][k][l])
'''
with torch.no_grad():
    detections = model(img)
    detections = non_max_suppression(detections,
                                     conf_thres,
                                     nms_thres)
    print(detections)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
img = np.array(Image.open(img_path))

fig, ax = plt.subplots(1)
ax.imshow(img)
if detections is not None:
# Rescale boxes to original image
    detections = rescale_boxes(detections[0], image_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        print("\t+ Label: %s, Conf: %.5f" % 
			(classes[int(cls_pred)], cls_conf.item()))
        box_w = x2 - x1
        box_h = y2 - y1
        color = bbox_colors[int(np.where(unique_labels
                                              == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1),
                                          box_w, 
                                          box_h, 
                                          linewidth=2,
                                          edgecolor=color, 
                                          facecolor="none")
                # Add the bbox to the plot
        ax.add_patch(bbox)
                # Add label
        plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

plt.axis("off")
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())
plt.savefig(f"./output.png", bbox_inches="tight", pad_inches=0.0)
plt.close()
