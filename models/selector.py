from models.lenet import *
from models.wresnet import *
import os

def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None):

    if dataset in ['SVHN', 'CIFAR10', 'CINIC10', 'CIFAR100']:
        n_classes = 100 if dataset == 'CIFAR100' else 10
        assert model_name in ['LeNet', 'WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2']
        if model_name=='LeNet':
            model = LeNet32(n_classes=n_classes)
        elif model_name=='WRN-16-1':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name=='WRN-16-2':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        elif model_name=='WRN-40-1':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name=='WRN-40-2':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)

        if pretrained:
            model_path = os.path.join(pretrained_models_path, dataset, model_name, "last.pth.tar")
            print('Loading Model from {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

    elif dataset=='ImageNet':
        assert model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        if model_name == 'ResNet18':
            model = resnet18(pretrained=pretrained)
        elif model_name == 'ResNet34':
            model = resnet34(pretrained=pretrained)
        elif model_name == 'ResNet50':
            model = resnet50(pretrained=pretrained)
        elif model_name == 'ResNet101':
            model = resnet101(pretrained=pretrained)
        elif model_name == 'ResNet152':
            model = resnet152(pretrained=pretrained)

    else:
        raise NotImplementedError

    return model

if __name__ == '__main__':

    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1))

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))