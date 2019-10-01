import torch
from solver import FewShotKTSolver
from utils.helpers import *

def main(args):
    """ Run several seeds """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if len(args.seeds) > 1:
        test_accs = []
        base_name = args.experiment_name
        for seed in args.seeds:
            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            set_torch_seeds(seed)
            args.experiment_name = os.path.join(base_name, base_name+'_seed'+str(seed))
            solver = FewShotKTSolver(args)
            test_acc = solver.run()
            test_accs.append(test_acc)
        mu = np.mean(test_accs)
        sigma = np.std(test_accs)
        print('\n\nFINAL MEAN TEST ACC: {:02.2f} +/- {:02.2f}'.format(mu, sigma))
        file_name = "mean_final_test_acc_{:02.2f}_pm_{:02.2f}".format(mu, sigma)
        with open(os.path.join(args.log_directory_path, base_name, file_name), 'w+') as f:
            f.write("NA")
    else:
        set_torch_seeds(args.seeds[0])
        solver = FewShotKTSolver(args)
        test_acc = solver.run()
        print('\n\nFINAL TEST ACC RATE: {:02.2f}'.format(test_acc))
        file_name = "final_test_acc_{:02.2f}".format(test_acc)
        with open(os.path.join(args.log_directory_path, args.experiment_name, file_name), 'w+') as f:
            f.write("NA")


if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    from utils.helpers import str2bool
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)  # For ImageNet
    print('Running...')

    parser = argparse.ArgumentParser(description='Welcome to BELS')

    parser.add_argument('--dataset', type=str, default='SVHN', choices=['SVHN', 'CIFAR10'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_images_per_class', type=int, default=100, help='=M in paper, -1 for all images')
    parser.add_argument('--scale_n_iters', type=float, default=1, help='For SVHN and low M lower n_iters doesnt hurt perf')
    parser.add_argument('--KD_alpha', type=float, default=0.9)
    parser.add_argument('--KD_temperature', type=float, default=4)
    parser.add_argument('--AT_beta', type=int, default=1000, help='beta for AT')
    parser.add_argument('--KT_mode', type=str, default='KD', choices=['KD', 'AT', 'KD+AT'], help='type of knowledge transfer loss')
    parser.add_argument('--pretrained_models_path', type=str, default='/home/paul/Pretrained/')
    parser.add_argument('--teacher_architecture', type=str, default='LeNet', help='use LeNet to debug on cpu')
    parser.add_argument('--student_architecture', type=str, default='LeNet', help='use LeNet to debug on cpu')
    parser.add_argument('--datasets_path', type=str, default="/home/paul/Datasets/Pytorch/")
    parser.add_argument('--log_directory_path', type=str, default="/home/paul/git/FewShotKnowledgeTransfer/logs/")
    parser.add_argument('--save_final_model', type=str2bool, default=False)
    parser.add_argument('--save_n_checkpoints', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default="/home/paul/git/FewShotKnowledgeTransfer/logs/")
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    args = parser.parse_args()

    if args.dataset == 'SVHN':
        n_classes = 10
        n_images = 73257
        n_base_epochs = 100
        n_base_lr_decay_steps = [30, 60, 80]
    elif args.dataset == 'CIFAR10':
        n_classes = 10
        n_images = 50e3
        n_base_epochs = 200
        n_base_lr_decay_steps = [60, 120, 160]
    else:
        raise NotImplementedError

    assert args.n_images_per_class*n_classes < n_images
    scale = args.scale_n_iters*n_images / (n_classes*args.n_images_per_class)
    args.n_epochs = n_base_epochs if args.n_images_per_class < 0 else int(n_base_epochs*scale)
    args.lr_decay_steps = n_base_lr_decay_steps if args.n_images_per_class < 0 else [int(i*scale) for i in n_base_lr_decay_steps]
    args.log_freq = int(args.n_epochs / 100)
    if 'AT' in args.KT_mode: assert args.student_architecture[:3] in args.teacher_architecture # need same architecture types for AT
    print('Epochs: {}, lr decays: {}'.format(args.n_epochs, args.lr_decay_steps))

    args.dataset_path = os.path.join(args.datasets_path, args.dataset)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    args.experiment_name = 'FewShotKnowledgeTransfer_{}_ne{}_{}_{}_{}_lr{}_bs{}_M{}_T{}_a{}_b{}'.format(args.dataset, args.n_epochs, args.teacher_architecture, args.student_architecture, args.KT_mode, args.learning_rate, args.batch_size, args.n_images_per_class, args.KD_temperature, args.KD_alpha, args.AT_beta)
    print('Logging results every {} epochs'.format(args.log_freq))
    print('\nRunning on device: {}'.format(args.device))

    main(args)
