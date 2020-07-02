import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/versa/dyy/pretrained_models/tf_efficientnet_lite3-b733e338.pth')
    state_dict = OrderedDict()
    for i, (k, v) in enumerate(net.items()):
        # print(i, k, v.shape)
        if 'classifier' in k:
            continue
        elif 'bn' in k:
            state_dict[k] = v
        else:
            k = k.replace(k.split('.')[-1], 'conv.' + k.split('.')[-1])
            state_dict[k] = v

    state_dict2 = OrderedDict()
    for i, (k, v) in enumerate(state_dict.items()):
        if 7 <= i <= 11:
            k = k.replace('bn1', 'bn2')
        elif i == 12:
            k = k.replace('conv_pw', 'conv_pwl')
        elif 13 <= i <= 17:
            k = k.replace('bn2', 'bn3')

        state_dict2[k] = v
        print(i, k, v.shape)
    torch.save(state_dict2, '/versa/dyy/pretrained_models/tf_efficientnet_lite3_modifed.pth')
