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
from pt import *


input_name = 'input0'
input_shape = [1]
#dtype = 'int32'
def convert_to_tvm():
    device = torch.device('cpu')
    model = torch.load("./a.pb", map_location=device)
    model = model.eval()

    input_data = torch.zeros(input_shape, dtype=torch.int)

    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    print(mod["main"])
    target = 'llvm'
    target_host = 'llvm'
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    return m, params

m, params = convert_to_tvm()
data = tvm.nd.array(np.zeros(input_shape, dtype=np.float32))
m.set_input(input_name, data)
m.set_input(**params)
m.run()
o = m.get_output(0).asnumpy()
print(o)

