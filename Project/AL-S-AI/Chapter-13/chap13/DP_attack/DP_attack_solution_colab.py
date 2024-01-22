[ ]
 import numpy as np
import os
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as Data
import copy as cp
import torch
import torch.optim as optim
!cp "/content/drive/MyDrive/Colab Notebooks/data_poisoning/resnet.py" .
from resnet import ResNet18
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def lr_scheduler(epoch):
    lr = 1e-3
    if epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)

    return lr

def test(net, Test_loader):

    net.eval()
    # test accuracy
    total = 0
    correct = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(Test_loader):  # for each training step
            # TODO: 
            # feed batch_x into the neural network
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = net(batch_x)
            _, preds = torch.max(outputs, dim=1)
            # compute test accuracy based on the output and batch_y
            total += len(batch_x)
            correct += (preds == batch_y).sum().item()
    print(f"test acc: {correct / total}")

def train(epoch, net, Train_loader):

    # Optimizer:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_scheduler(epoch)) # MNIST

    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # training
    loss_train = 0
    total = 0
    correct = 0

    net.train()
    # t = tqdm(total=len(Train_loader.dataset), unit="samples")
    for step, (batch_x, batch_y) in enumerate(Train_loader):  # for each training step

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device, dtype=torch.long)

        # TODO:
        # Question: do I need to "zero the gradients" for this batch? was done in 
    # the tutorial on this site: 
    # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        # optimizer.zero_grad()

        # feed batch_x into the neural network
        outputs = net(batch_x)
        _, preds = torch.max(outputs, dim=1)    # predictions
        # compute loss, update parameters of the neural network
        loss = loss_func(outputs, batch_y)
        loss.backward() # Question: what exactly does "backward()" do?
        optimizer.step()
        # compute training accuracy based on the output and batch_y
        total += len(preds)
        correct += (preds == batch_y).sum().item()
        # t.update(len(batch_x))
    loss_train = loss_train / (step + 1)
    train_acc = correct / total
    print("{}, avg_train_loss: {}, train_acc: {}".format(epoch, loss_train, train_acc))
    return net

def DataPoisoning(N0, N1, N2, N3, N4):
    # Ni: # malicious samples from class i
    # cats = [0,1,2,3,4]: 5 classes of MNIST

    Attack = [N0, N1, N2, N3, N4]

    # 1.load datasets
    # ===============
    # transforms.Compose is a class used for chaining together 
  # torchvision.transforms (transforms are common transformations done on images).
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_path = "/content/drive/MyDrive/Colab Notebooks/data_poisoning/dataset"
    trainset = torchvision.datasets.MNIST(
            root= dataset_path, 
            train=True, download=True, transform=transform_train
            )
    testset = torchvision.datasets.MNIST(
            root= dataset_path,
            train=False, download=True, transform=transform_test
            )
    attackset = cp.copy(trainset)

    # 2. the experiments only involve 5 classes of MNIST [0, 1, 2, 3, 4]
    # split the training set, 2000 samples per class for training, 1000 samples 
  # per class for testing, 800 samples per class for poisoning
    cats = [0, 1, 2, 3, 4]
    NC = len(cats)
    nTrain = 2000
    nTest = 1000
    NAttack = 800

    # labels are stored in a list in datasets, converted from a tensor to a numpy array
    trainset.targets = np.array(trainset.targets)
    attackset.targets = np.array(attackset.targets)
    testset.targets = np.array(testset.targets)
    # store indices of training samples and malicious samples
    ind_training = np.array([], dtype=int)
    ind_attack = np.array([], dtype=int)
    for c in cats:
        # 1) randomly select NTrain+NAttack training samples from class c in [0,1,2,3,4]
        ind_c = np.where(trainset.targets == c)[0]  # this returns a list of indices where trainset.targets[i] = c
        ind_c = np.random.choice(ind_c, size=nTrain+NAttack, replace=False) # this randomly selects nTrain+NAttack sample indices from the list returned on the previous line
        # 2) Use a mask to split training samples and malicious samples.
        # The sample added to ind_training and ind_attack are both of class c, but we need to split this so that we can then change the class of the samples in ind_attack to use them later for poisoning
        mask = np.ones(ind_c.shape, dtype=bool)
        mask[:NAttack] = False
        # NTrain samples are used for training, NAttack samples are used for poisoning
        ind_training = np.concatenate((ind_training, ind_c[mask]))
        ind_attack = np.concatenate((ind_attack, ind_c[np.logical_not(mask)]))
        
    # extract samples by indices
    trainset.targets = trainset.targets[ind_training]   # trainset.targets.size = 10000 (`nTrain` samples from each class)
    trainset.data = trainset.data[ind_training]         # trainset.data.shape = torch.Size([10000, 28, 28])

    # attackset.targets.size = 4000 (`NAttack` samples from each class)
    attackset.targets = attackset.targets[ind_attack]
    attackset.data = attackset.data[ind_attack]         # attackset.data.size = torch.Size([4000, 28, 28])

    # store indices of test samples
    ind_test = np.array([], dtype=int)
    # only keep test samples from the 5 classes
    for c in cats:
        ind_c = np.where(testset.targets == c)[0]
        ind_test = np.concatenate((ind_test, ind_c))
    testset.targets = testset.targets[ind_test]
    testset.data = testset.data[ind_test]

    # 3. poison the training set
    # Attack=[N0,N1,N2,N3,N4]
    # if Ni==1, then evenly distribute samples from class i of the attack set to the training sets of classes c!=i
    """Questions:
    - What does it mean to "evenly distribute" the samples into the training sets?
        - I'm going to try this by concatenating the dataset w/ the poisend images and then shuffling the data
    - The training sets of all classes are all stored in trainset.data right? Where each consecutive set of `nTrain` samples have the same target class?
    - How big should trainset.data and trainset.targets be after this for loop executes?
    """
    for j, attack in enumerate(Attack):
        if attack != 0:
            # select samples of cats[j] from the attackset
                # Question: I'm assuming this means I should use all the samples of class cats[j] in the attackset
            i_j = np.where(attackset.targets == j)[0]
            # mislabel those samples as c != cats[j], that is,
            # for each class c!=cats[j], put NAttack/(len(Attack)-1) labels of c in trainset.targets
            attack_data_j = attackset.data[i_j]
            attack_targets_j = attackset.targets[i_j]

            poison_labels = set(Attack) - set([j])
            L = 0
            R = int(NAttack / (len(Attack) - 1))
            for p in poison_labels:
                attack_targets_j[L:R] = p
                L += int(NAttack / (len(Attack) - 1))
                R += int(NAttack / (len(Attack) - 1))
            # put those samples into the training set
            trainset.data = torch.cat((trainset.data, attack_data_j))
            trainset.targets = np.concatenate((trainset.targets, attack_targets_j))

    # Here I shuffle the data to mix the attackset samples in with the trainset
    def shuffle(arr1, arr2):
        assert(len(arr1) == len(arr2))
        perm = np.random.permutation(len(arr1))
        return arr1[perm], arr2[perm]
    
    trainset.data, trainset.targets = shuffle(trainset.data, trainset.targets)

    # 4. form dataloader for CNN
    BATCH_SIZE = 32 
    EPOCH = 50
    Train_loader = Data.DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=1, )

    Test_loader = Data.DataLoader(
        dataset=testset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=1, )

    # 5. train ResNet on the poisoned training set
    # TODO: implement code for ResNet training and testing
    # function train(epoch, net, Train_loader)
    # function test(epoch, net, Test_loader)
    net = ResNet18(num_classes=NC)
    net.to(device)
    for epoch in range(EPOCH):
        net = train(epoch, net, Train_loader)
        test(net, Test_loader)
    # save model
    model_path = f'/content/drive/MyDrive/Colab Notebooks/data_poisoning/models/poisoned_{N0}_{N1}_{N2}_{N3}_{N4}.pth'
    torch.save(net.state_dict(), model_path)


