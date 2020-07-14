import torch
import torch.nn as nn
import numpy as np

device="cpu"
class simple_net(nn.Module):
    def __init__(self):
        super(simple_net, self).__init__()
        self.a = torch.from_numpy(np.array((11.0,)).astype('float32'))
        self.b = torch.from_numpy(np.array((2.0,)).astype('float32'))

    def forward(self, inputs):
        self.b.copy_(self.a)
        print("forward")
        #self.b = torch.clone(self.a)
        output = torch.add(inputs,self.b)
        return output

m = simple_net()
input_data = torch.zeros((1))
m.forward(input_data)
print("m.b is"+ str(m.b))
m = m.eval()
params = list(m.parameters())
print(params)
m.to(device)
print(m.eval())
torch.save(m, "./a.pb")


