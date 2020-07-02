import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/home/dingyangyang/SOLO/work_dirs/solov2_attention_label_align2_assim/stage2_epoch_8_0.399.pth')
    state_dict = OrderedDict()
    for k, v in net['state_dict'].items():
        # print(k, v.shape)
        if k.startswith('backbone') or k.startswith('neck'):
            state_dict[k] = v

    for k, v in state_dict.items():
        print(k)
    torch.save(state_dict, '/home/dingyangyang/SOLO/work_dirs/solov2_attention_label_align2_assim/0.399_rm_head.pth')
