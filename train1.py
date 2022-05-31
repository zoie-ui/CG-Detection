import math
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import json
import time
import warnings
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from PIL import Image
from PIL import ImageFile
from DualNet import DualNet
import pdb

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class Rgb2Gray(object):
    def __init__(self):
        '''rgb_image = object.astype(np.float32)
        if len(rgb_image.shape)!= 3 or rgb_image.shape[2] != 3:
            raise ValueError("input image is not a rgb image!")'''

    def __call__(self, img):
        L_image = img.convert('L')
        return L_image


data_transform = {
    "train": transforms.Compose(
        [transforms.CenterCrop(224),
         transforms.ToTensor()
         ]),
    "val": transforms.Compose(
        [transforms.CenterCrop(224),
         transforms.ToTensor()
         ]),
    "tensor": transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
}

image_path = "/data/CG/DsTok/raw"

train_dataset = datasets.ImageFolder(root=image_path + '/train',
                                     transform=data_transform['train']
                                     # transform1=data_transform['tensor']
                                     )
# print('index of class: ')
# print(train_dataset.class_to_idx)
# print(train_dataset.imgs[0][0])

train_num = len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=True,
                                           drop_last=True)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform['train']
                                        # transform1=data_transform['tensor']
                                        )
val_num = len(validate_dataset)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              drop_last=True)

test_dataset = datasets.ImageFolder(root=image_path + '/test',
                                    transform=data_transform['val']
                                    # transform1=data_transform['tensor']
                                    )
test_num = len(test_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=64,
                                          shuffle=True,
                                          drop_last=False)
# print(xception_modules)
save_path = './result/pth/DsTok_raw_DualNet.pth'
save_txt = './result/txt/DsTok_raw_DualNet.txt'
def train():
    net = DualNet()
    net = nn.DataParallel(net)
    net = net.cuda()

    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    best_acc = 0.0
    new_lr = 1e-3
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, 120):
        if epoch % 20 == 0:
            new_lr = new_lr * 0.5
            optimizer = optim.SGD(net.parameters(), lr=new_lr, momentum=0.9, weight_decay=1e-3)
        net.train()
        running_loss = 0.0
        running_corrects = 0.0
        time_start = time.perf_counter()
        warnings.filterwarnings('ignore')
        # count = 0
        # if epoch % 50 == 0:
        print('epoch[%d]  ' % epoch)

        for step, data in enumerate(train_loader, start=0):
            images, labels = data  #
            optimizer.zero_grad()
            outputs = net(images.cuda())
            preds = torch.max(outputs, dim=1)[1]
            loss = loss_function(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += (preds == labels.cuda()).sum().item()

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 20)
            b = "." * int((1 - rate) * 20)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end='')
        print()
        print('%f s' % (time.perf_counter() - time_start))
        print('train_loss: %.3f   train_acc: %.3f' % (running_loss / step, running_corrects / train_num))
        f = open(save_txt, mode='a')
        f.write("epoch[{:.0f}]\ntrain_loss:{:.3f}    train_acc:{:.3f}\n".format(epoch, running_loss / step,
                                                                                running_corrects / train_num))
        f.close()

        ########################################### validate ###########################################

        # net = torch.load('Result2.pth')
        val_loss = 0.0
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for step, val_data in enumerate(validate_loader, start=0):
                val_images, val_labels = val_data
                outputs= net(val_images.cuda())
                loss1 = loss_function(outputs, val_labels.cuda())
                result = torch.max(outputs, dim=1)[1]

                val_loss += loss1.item()
                acc += (result == val_labels.cuda()).sum().item()
            val_accurate = acc / val_num
            print('val_loss: %.3f  val_acc: %.3f' % (val_loss / step, val_accurate))
            # print()
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            f = open(save_txt, mode='a')
            f.write("val_loss:{:.3f}    val_accurate:{:.3f}\n".format(val_loss / step, val_accurate))
            f.close()
    print('best acc: %.3f' % best_acc)
    f = open(save_txt, mode='a')
    f.write("  best_acc:{:.3f}".format(best_acc) + '\n')
    f.close()
    print('Finished Training')

def test():
    net = DualNet()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(save_path))
    net.cuda()
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data
            outputs = net(test_images.cuda())
            result = torch.max(outputs, dim=1)[1]
            acc += (result == test_labels.cuda()).sum().item()
            # print('acc of every batch is: %.3f' % (acc/32))
        test_accurate = acc / test_num
        f = open(save_txt, mode='a')
        f.write("  test_acc:{:.3f}".format(test_accurate) + '\n')
        f.close()
        print('acc of val is: %.3f' % (test_accurate))

if __name__ == '__main__':
    train()
    test()
