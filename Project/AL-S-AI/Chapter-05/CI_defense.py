"""
Cluster Impurity (CI) defense:
Author: Zhen Xiang
Date: 7/26/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import copy as cp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from CNN.resnet import ResNet18


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def im_process(images, process_type, process_parameter):

    if process_type == 'noise':
        images += torch.randint(-int(process_parameter), int(process_parameter + 1), images.size()).to(device) / 255.0
        images.clamp(0, 1)
    if process_type == 'avg_filter':
        if ~int(process_parameter) % 2:
            temp = torch.zeros((images.size(0), images.size(1), images.size(2) + 1, images.size(3) + 1)).to(device)
            temp[:, :, :images.size(2), :images.size(3)] = images
            images = temp
        padding_size = int((process_parameter - 1) / 2)
        weight = torch.zeros(images.size(1), images.size(1), int(process_parameter), int(process_parameter)).to(device)
        for c in range(images.size(1)):
            weight[c, c, :, :] = torch.ones((int(process_parameter), int(process_parameter))) / (process_parameter ** 2)
        images = F.conv2d(images, weight, padding=padding_size)
    if process_type == 'med_filter':
        if ~int(process_parameter) % 2:
            temp = torch.zeros((images.size(0), images.size(1), images.size(2) + 1, images.size(3) + 1)).to(device)
            temp[:, :, :images.size(2), :images.size(3)] = images
            images = temp
        padding_size = int((process_parameter - 1) / 2)
        temp = torch.zeros(
            (images.size(0), images.size(1), images.size(2) + 2 * padding_size, images.size(3) + 2 * padding_size)).to(
            device)
        temp[:, :, padding_size:images.size(2) + padding_size, padding_size:images.size(3) + padding_size] = images
        images = temp
        images_unfolded = images.unfold(2, int(process_parameter), 1).unfold(3, int(process_parameter), 1)
        images_unfolded = torch.reshape(images_unfolded, (
        images_unfolded.size(0), images_unfolded.size(1), images_unfolded.size(2), images_unfolded.size(3), -1))
        images = torch.median(images_unfolded, -1)[0]
    if process_type == 'quant':
        l = 2 ** process_parameter
        images = (torch.round(images * l + 0.5) - 0.5) / l

    return images


class Hook():
    '''
    For Now we assume the input[0] to last linear layer is a 1*d tensor
    the layerOutput is a list of those tensor value in numpy array
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.layerOutput = None

    def hook_fn(self, module, input, output):
        feature = input[0].cpu().numpy()
        if self.layerOutput is None:
            self.layerOutput = feature
        else:
            self.layerOutput = np.append(hooker.layerOutput, feature, axis=0)
        pass

    def close(self):
        self.hook.remove()


def getLayerOutput(ds, model, hook, ic=None, batch_size=128):
    ''' Get the layer outputs
    Args:
        ds (torch.tensor): dataset of data
        model (torch.module):
        hook (Hook): self-defined hook class
        ind_correct (np.array): record the indices of samples correctly classified
        outs (None/np.array): record  nn models' ouput (num_samples, class_nums)
            if none, no recording
    Returns: None
    '''
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)
    model.eval()
    correct = 0
    tot = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            tot += targets.size(0)

            if ic is not None:
                ic = torch.cat((ic, predicted.eq(targets).nonzero().squeeze()))

    hook.close()
    print('acc: {}/{} = {:.2f}'.format(correct, tot, correct / tot))
    return ic


transform_train = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# Load in backdoor training images
if not os.path.isdir('attacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./attacks/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
attackset = torch.utils.data.TensorDataset(train_images_attacks, train_labels_attacks)

# Delete training images used for backdoor training)
ind_train = torch.load('./attacks/ind_train')
trainset.data = np.delete(trainset.data, ind_train, axis=0)
trainset.targets = np.delete(trainset.targets, ind_train, axis=0)

model = ResNet18()
model.load_state_dict(torch.load('./contam/model_contam.pth'))
model = model.to(device)
model = model.eval()

targetlayer = model._modules['linear']
layer_name = 'linear'
if not os.path.isdir('features'):
    os.mkdir('features')
np.save('./features/layer_name', layer_name)

# Extract features for clean training images
hooker = Hook(targetlayer)
ind_correct = torch.tensor([], dtype=torch.long).to(device)
ind_correct = getLayerOutput(trainset, model, hooker, ind_correct, batch_size=32)
feature_clean = hooker.layerOutput
# Categorize features based on labels
NC = np.max(trainset.targets)+1
for c in range(NC):
    ind = [i for i, label in enumerate(trainset.targets) if label == c]
    np.save('./features/feature_{}'.format(str(c)), feature_clean[ind, :])
    np.save('./features/ind_{}'.format(str(c)), ind)

# Extract features for backdoor training images
hooker = Hook(targetlayer)
getLayerOutput(attackset, model, hooker, None, batch_size=32)
feature_attack = hooker.layerOutput
np.save('./features/feature_attack', feature_attack)
np.save('./features/attack_TC', train_labels_attacks[0])


# CI parameters
thres = 0.5

X1 = feature_attack
TC = train_labels_attacks[0]

TP_count = 0
FP_count = 0
clean_total = 0
attack_total = X1.shape[0]
detection_flag = False
target_correct = False
remove_ind = None
remove_ind_attack = None

# Clustering
for c in range(NC):
    print('processing class {}'.format(c))
    X = np.load('./features/feature_{}.npy'.format(str(c)))
    if c == TC:
        X = np.concatenate((X, X1))
    ind = np.load('./features/ind_{}.npy'.format(str(c)))
    clean_total += ind.shape[0]
    ascent_count = 0
    BIC_best = float('inf')
    model_best = None
    n_comp = 1
    while ascent_count < 2:
        model = GaussianMixture(n_components=n_comp,
                                covariance_type='full',
                                reg_covar=1e-3,
                                max_iter=100, init_params='kmeans',
                                n_init=1,
                                )
        model.fit(X)
        BIC = model.bic(X)
        if BIC < BIC_best:
            BIC_best = BIC
            model_best = model
            ascent_count = 0
        else:
            ascent_count += 1

        n_comp += 1

    # If there is a single component, move on to the next class
    if len(model_best.weights_) == 1:
        continue

    # Blur the image and get the impurity
    trainclass = cp.copy(trainset)
    trainclass.data = trainclass.data[ind]
    trainclass.targets = trainclass.targets[ind]
    trainclassloader = torch.utils.data.DataLoader(trainclass, batch_size=32, shuffle=False, num_workers=2)
    impurity_indicator = torch.tensor([], dtype=torch.bool)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainclassloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            outputs_blurred = model(im_process(inputs, process_type='avg_filter', process_parameter=2))
            _, predicted_blurred = outputs_blurred.max(1)
            predicted, predicted_blurred = predicted.cpu(), predicted_blurred.cpu()
            impurity_indicator = torch.cat((impurity_indicator, torch.eq(predicted, predicted_blurred)))
    if c == TC:
        attackloader = torch.utils.data.DataLoader(attackset, batch_size=32, shuffle=False, num_workers=2)
        impurity_indicator_attack = torch.tensor([], dtype=torch.bool)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(attackloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                outputs_blurred = model(im_process(inputs, process_type='avg_filter', process_parameter=2))
                _, predicted_blurred = outputs_blurred.max(1)
                predicted, predicted_blurred = predicted.cpu(), predicted_blurred.cpu()
                impurity_indicator_attack = torch.cat((impurity_indicator_attack, torch.eq(predicted, predicted_blurred)))

    comp_label = model_best.predict(X)
    if c == TC:
        comp_label_attack = comp_label[len(comp_label)-attack_total:]
        comp_label = comp_label[:len(comp_label)-attack_total]
    # Get CI for each component
    for m in range(len(model_best.weights_)):
        ind_class = [i for i, lb in enumerate(comp_label) if lb == m]
        comp_size = len(ind_class)
        CI_count = torch.sum(impurity_indicator[ind_class]).numpy()
        if c == TC:
            ind_class_attack = [i for i, lb in enumerate(comp_label_attack) if lb == m]
            comp_size += len(ind_class_attack)
            CI_count += torch.sum(impurity_indicator_attack[ind_class_attack]).numpy()
        CI = -np.log(CI_count / comp_size)
        print('CI for component {}: {}'.format(m, CI))
        if CI > thres:
            detection_flag = True
            FP_count += len(ind_class)
            if remove_ind is None:
                remove_ind = ind[ind_class]
            else:
                remove_ind = np.concatenate((remove_ind, ind[ind_class]))
            if c == TC:
                target_correct = True
                TP_count += len(ind_class_attack)
                if remove_ind_attack is None:
                    remove_ind_attack = ind_class_attack
                else:
                    remove_ind_attack = np.concatenate((remove_ind_attack, ind_class_attack))

TPR = TP_count / attack_total
FPR = FP_count / clean_total

if not os.path.isdir('CI_results'):
    os.mkdir('CI_results')

if not detection_flag:
    print('No attack detected -- a failure')
else:
    print('Attack detected!')
    if not target_correct:
        print('Target class incorrectly inferred!')
    else:
        print('Target class correctly inferred!')
    print('TPR: {}; FPR: {}'.format(TPR, FPR))

np.save('./CI_results/remove_ind', remove_ind)
np.save('./CI_results/remove_ind_attack', remove_ind_attack)
