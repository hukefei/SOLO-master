import torch
from collections import OrderedDict

net = torch.load('/versa/dyy/SOLO/solo2-8900b877.pth')

state_dict = OrderedDict()
for k, v in net['state_dict'].items():
    print(k)
    if 'gn' in k:
        k = k.replace('gn', 'bn')
        state_dict[k] = v
        if 'bias' in k:
            k1 = k.replace('bias', 'running_mean')
            state_dict[k1] = torch.zeros(160)
            k2 = k.replace('bias', 'running_var')
            state_dict[k2] = torch.ones(160)
    else:
        state_dict[k] = v

model = {'meta': net['meta'], 'state_dict': state_dict}
torch.save(model, '/versa/dyy/SOLO/solo2_gn2bn.pth')

# net = torch.load('/versa/dyy/SOLO/decoupled_solo-8f2efb80.pth')
#
# state_dict = OrderedDict()
# for k, v in net['state_dict'].items():
#     print(k)
#     if 'gn' not in k:
#         state_dict[k] = v
#
# torch.save(state_dict, '/versa/dyy/SOLO/decoupled_rm_gn.pth')