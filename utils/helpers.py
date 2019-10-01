import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import argparse
import os

class AggregateScalar(object):
    """
    Computes and stores the average and std of stream.
    Mostly used to average losses and accuracies.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0001  # DIV/0!
        self.sum = 0

    def update(self, val, w=1):
        """
        :param val: new running value
        :param w: weight, e.g batch size
        """
        self.sum += w * (val)
        self.count += w


    def avg(self):
        return self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target=torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1/batch_size))
    return res

def plot_image(input):
    if input.shape[0]==1:
        input = torch.cat([input, input, input])
    npimg = np.transpose(input.numpy(), (1, 2, 0))
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.imshow(npimg, cmap='gray')
    plt.show()
    return fig

def plot_images(batch, ncol=None):
    n_images = batch.shape[0]
    ncol = ncol if ncol is not None else int(np.ceil(np.sqrt(n_images)))
    grid_img = torchvision.utils.make_grid(batch,
                                           nrow=ncol,
                                           padding=2,
                                           pad_value=0)
    # black and white images converted to color
    plot_image(grid_img)

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def delete_files_from_name(folder_path, file_name, type='contains'):

    assert type in ['is', 'contains']
    for f in os.listdir(folder_path):
        if (type=='is' and file_name==f) or (type=='contains' and file_name in f):
            os.remove(os.path.join(folder_path, f))