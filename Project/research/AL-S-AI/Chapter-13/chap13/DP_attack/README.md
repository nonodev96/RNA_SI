# Data poisoning attack

Goal: Decrease the test accuracy of ResNet-18 on MNIST
Method: Poison the training set of MNIST by mislabelling images. For example, add 100 images of class 0 to class 1 and label them as class 1.
Experiments:

1. Load training and test sets of MNIST
2. Keep training and test samples of 5 classes [0,1,2,3,4]. For each class, split the training set: 2000 images are used for training, 800 images are used for poisoning. Poisoning samples are stored in another dataset named attackset.
3. (TODO) poison the training set. There are 5 inputs, N0,N1,N2,N3,N4. If Nc is 1, then evenly distribute poisoning samples of class c into the training sets of the remaining 4 classes. For example, if N0=1, add 200 poisoning samples of class 0 into class 1, and label them as class 1. Do the same to poison class 2,3,4 by class 0.
4. (TODO) train a ResNet-18 on poisoned training set.  
5. (TODO) test the trained ResNet-18 on clean (attack-free) test set.
For 4 and 5, the frameworks of train(epoch, net, Train_loader) and test(net, Test_loader) are given. The learning rate, the number of training epochs, the loss function and the optimizer are given. You need to implement forward and backward propagation.
6. Do the experiment 6 times. In experiment j, use j classes for poisoning. The number of classes used for poisoning is the attack strength. For example, in experiment 0, N0=0, N1=0, N2=0, N3=0, N4=0, there’s no poisoning (this is the control experiment). In experiment 3, N0=1, N1=1, N2=1, N3=0, N4=0, 3 classes are used for poisoning, and all the 5 classes are poisoned. Observe the test accuracy vs the attack strength. The stronger the attack, the lower the test accuracy.
