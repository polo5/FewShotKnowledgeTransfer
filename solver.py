import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import shutil
from time import time
from utils.logger import *
from utils.datasets import *
from models.selector import *


class FewShotKTSolver(object):
    def __init__(self, args):
        self.args = args
        self.teacher = select_model(dataset = args.dataset,
                                    model_name = args.teacher_architecture,
                                    pretrained = True,
                                    pretrained_models_path = args.pretrained_models_path).to(device=args.device)
        self.student = select_model(dataset = args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path = args.pretrained_models_path).to(device=args.device)
        self.teacher.eval()

        self.optimizer = optim.SGD(self.student.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)  # wresnet configs
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_decay_steps, gamma=0.2)  # wresnet configs

        ### Set up & Resume
        self.start_epoch = 0
        self.experiment_path = os.path.join(args.log_directory_path, args.experiment_name)
        self.save_model_path = os.path.join(args.save_model_path, args.experiment_name)
        self.logger = Logger(log_dir=self.experiment_path)
        self.indices = None

        if os.path.exists(self.experiment_path):
            if self.args.use_gpu:
                checkpoint_path = os.path.join(self.experiment_path, 'last.pth.tar')
                if os.path.isfile(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    print('\nResuming from checkpoint file at epoch {} with top 1 acc {}\n'.format(checkpoint['epoch'], checkpoint['test_acc1']))
                    self.start_epoch = checkpoint['epoch']
                    self.student.load_state_dict(checkpoint['state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.indices = checkpoint['indices']
            else:
                ## clear debug logs on cpu
                shutil.rmtree(self.experiment_path)
                os.makedirs(self.experiment_path)
        else:
            os.makedirs(self.experiment_path)

        ## Get loaders
        self.train_loader, self.test_loader, self.indices = get_loaders(args, self.indices)

        ## Save args
        print('\n---------')
        with open(os.path.join(self.experiment_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

    def run(self):

        for epoch in range(self.start_epoch, self.args.n_epochs):

            train_acc, train_KT_loss, data_time, batch_time = self.train()
            self.scheduler.step(epoch=epoch)

            if epoch % self.args.log_freq == 0:
                test_acc, test_KT_loss = self.test()
                print('\nEpoch {}/{} -- Train Acc: {:02.2f}% -- Train KT Loss: {:02.2f}'.format(epoch, self.args.n_epochs, train_acc*100, train_KT_loss))
                #print('Test Acc: {:02.2f}% -- Test KT Loss: {:02.2f}'.format(test_acc*100, test_KT_loss))
                self.logger.scalar_summary('TRAIN/acc', train_acc*100, epoch)
                self.logger.scalar_summary('TRAIN/KT_loss', train_KT_loss, epoch)
                self.logger.scalar_summary('TIME/data_time_sec', data_time, epoch)
                self.logger.scalar_summary('TIME/batch_time_sec', batch_time, epoch)
                self.logger.scalar_summary('TEST/acc', test_acc*100, epoch)
                self.logger.scalar_summary('TEST/KT_loss', test_KT_loss, epoch)
                self.logger.write_to_csv('train_test.csv')
                self.logger.writer.flush()

            if self.args.save_n_checkpoints > 1:
                if (epoch+1) % int(self.args.n_epochs / self.args.save_n_checkpoints) == 0:
                    test_acc, test_KT_loss = self.test()
                    self.save_model(epoch=epoch, test_acc=test_acc)

        test_acc, test_KT_loss = self.test()
        if self.args.save_final_model:  # make sure last epoch saved
            self.save_model(epoch=epoch, test_acc=test_acc)

        return test_acc * 100


    def train(self):
        self.student.train()
        data_time, batch_time = AggregateScalar(), AggregateScalar()
        running_acc, running_KT_loss = AggregateScalar(), AggregateScalar()

        end = time()
        for idx, (x, y) in enumerate(self.train_loader):
            data_time.update(time() - end)
            x, y = x.to(self.args.device, non_blocking=True), y.to(self.args.device, non_blocking=True)
            student_logits, *student_activations = self.student(x)
            teacher_logits, *teacher_activations = self.teacher(x)
            KT_loss = self.KT_loss(y, student_logits, student_activations, teacher_logits, teacher_activations)
            acc = accuracy(student_logits.data, y, topk=(1,))[0]
            running_acc.update(float(acc), x.shape[0])
            running_KT_loss.update(float(KT_loss), x.shape[0])

            self.optimizer.zero_grad()
            KT_loss.backward()
            self.optimizer.step()

            batch_time.update(time() - end)
            end = time()
            # print(idx, data_time.avg(), batch_time.avg())

        return running_acc.avg(), running_KT_loss.avg(), data_time.avg(), batch_time.avg()

    def test(self):
        self.student.eval()
        running_acc, running_KT_loss = AggregateScalar(), AggregateScalar()
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = self.student(x)
                teacher_logits, *teacher_activations = self.teacher(x)
                KT_loss = self.KT_loss(y, student_logits, student_activations, teacher_logits, teacher_activations)
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_KT_loss.update(float(KT_loss), x.shape[0])
                running_acc.update(float(acc), x.shape[0])

        return running_acc.avg(), running_KT_loss.avg()


    def attention(self, x):
        """:param x = activations"""
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """:param x,y = activations"""
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def KT_loss(self, true_labels, student_logits, student_activations, teacher_logits, teacher_activations):
        """ verbose on purpose for clarity """

        if self.args.KT_mode == 'KD':
            # Becomes cross entropy for alpha = 0
            student_CE = F.cross_entropy(student_logits, true_labels)
            student_teacher_KL = F.kl_div(F.log_softmax(student_logits / self.args.KD_temperature, dim=1), F.softmax(teacher_logits / self.args.KD_temperature, dim=1))
            loss = (1. - self.args.KD_alpha)*student_CE + (2 * self.args.KD_alpha * self.args.KD_temperature**2)*student_teacher_KL # 2*alpha -> alpha in official AT code

        elif self.args.KT_mode == 'AT':
            student_CE = F.cross_entropy(student_logits, true_labels)
            loss = student_CE
            adjusted_beta = (self.args.AT_beta * 3) / len(student_activations) #default given beta is for 3 attention terms
            for i in range(len(student_activations)):
                loss = loss + adjusted_beta * self.attention_diff(student_activations[i], teacher_activations[i])

        elif self.args.KT_mode == 'KD+AT':
            # In AT paper beta is decayed in that mode but judged an overkill here
            # This loss becomes = AT if alpha=0
            student_CE = F.cross_entropy(student_logits, true_labels)
            student_teacher_KL = F.kl_div(F.log_softmax(student_logits / self.args.KD_temperature, dim=1), F.softmax(teacher_logits / self.args.KD_temperature, dim=1))
            loss = (1. - self.args.KD_alpha) * student_CE + (2 * self.args.KD_alpha * self.args.KD_temperature ** 2) * student_teacher_KL  # 2*alpha -> alpha in official AT code
            adjusted_beta = (self.args.AT_beta * 3) / len(student_activations)
            for i in range(len(student_activations)):
                loss = loss + adjusted_beta * self.attention_diff(student_activations[i], teacher_activations[i])


        return loss


    def save_model(self, epoch, test_acc):

        delete_files_from_name(self.save_model_path, "test_acc_", type='contains')
        file_name = "epoch{}_test_acc_{:02.2f}".format(epoch, test_acc * 100)
        with open(os.path.join(self.save_model_path, file_name), 'w+') as f:
            f.write("NA")

        torch.save({'args': self.args,
                    'epoch': epoch + 1,
                    'state_dict': self.student.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'indices': self.indices,
                    'test_acc': test_acc},
                   os.path.join(self.save_model_path, "last.pth.tar"))
        print("\nSaved model with test acc {:02.2f}%\n".format(test_acc * 100))
