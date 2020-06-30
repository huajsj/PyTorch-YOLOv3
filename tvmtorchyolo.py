import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision

import argparse

from tvm import rpc, autotvm, relay
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tvm
import vta
from tvm import rpc, autotvm, relay
from tvm.relay.testing import yolo_detection
#from tvm.relay.testing.darknet import __darknetffi__
from tvm.contrib import graph_runtime, graph_runtime, util
from tvm.contrib.download import download_testdata
from vta.testing import simulator
from vta.top import graph_pack
from utils.datasets import *
from utils.utils import *


input_name = 'input0'
dtype = 'float32'
def convert_to_tvm(model_path, image_channel, image_size):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model = model.eval()

    input_shape = [1, image_channel, image_size, image_size]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)
    print(mod["main"])

    target = 'llvm'
    target_host = 'llvm'
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)
    from tvm.contrib import graph_runtime
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    return m, params

def convert_to_vta(model_path, image_channel, image_size):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model = model.eval()

    input_shape = [1, image_channel, image_size, image_size]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)
    print(mod["main"])
    
    remote = rpc.LocalSession()
    ctx = remote.ext_dev(0)

    target = 'vta'
    target_host = 'vta'
    env = vta.get_env()
    pack_dict = {
    "yolov3-tiny": ["nn.max_pool2d", "cast", 8, 237],
    }
    MODEL_NAME = 'yolov3-tiny'
    with tvm.transform.PassContext(opt_level=2):
            with relay.quantize.qconfig(global_scale=33.0,
                                        skip_conv_layers=[0],
                                        store_lowbit_output=True,
                                        round_for_shift=True):
                mod = relay.quantize.quantize(mod, params=params)
            print(mod["main"])
            mod = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=pack_dict[MODEL_NAME][0],
                stop_name=pack_dict[MODEL_NAME][1],
                start_name_idx=pack_dict[MODEL_NAME][2],
                stop_name_idx=pack_dict[MODEL_NAME][3])
    return mod

def process_image(img_path, image_size):
    img = Image.open(img_path).resize((image_size, image_size))
    from torchvision import transforms
    
    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img, _ = pad_to_square(img, pad_value=0)# trainning dataset logic
    img = resize(img, image_size)
    img = img.unsqueeze(0)
    return img.cpu().numpy()

def box_mark(img_path, detections):
    print(detections)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.ticker import NullLocator
    class_path = "data/coco.names"
    classes = load_classes(class_path)
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
    plt.show()
    plt.close()
    
def detect(m, params, img_path, image_size):
    from PIL import Image
    #img = Image.open(img_path).resize((image_size, image_size))
    #img = np.expand_dims(img, 0)
    img = process_image(img_path, image_size)

    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    m.set_input(**params)
    # Execute
    m.run()
    conf_thres = 0.5
    nms_thres = 0.4
    detections = torch.from_numpy(m.get_output(0).asnumpy())
    detections = non_max_suppression(detections,
                                     conf_thres,
                                     nms_thres)
    box_mark(img_path,detections)
    
    

model_path="./checkpoints/yolov3_tiny_all_40.pth"
image_size=320
image_channel=3
image_path="./person.jpg"

m, params = convert_to_vta(model_path,
                   image_channel,
                   image_size)

detect(m, params, image_path, image_size)
