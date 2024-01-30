#!/usr/bin/python3

import numpy as np
from scipy import sparse, stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import os
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as Data
import copy as cp
import torch
import torch.optim as optim
from resnet import ResNet18
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cats = [0, 1, 2, 3, 4]

def lr_scheduler(epoch):
	lr = 1e-3
	if epoch > 20:
		lr *= 1e-2
	elif epoch > 10:
		lr *= 1e-1
	print('Learning rate: ', lr)

	return lr

def test(net, Test_loader):

	net.eval()
	# test acc
	total = 0
	correct = 0
	with torch.no_grad():
		for step, (batch_x, batch_y) in enumerate(Test_loader):  # for each training step

			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device, dtype=torch.long)

			prediction = net(batch_x)  # input x and predict based on x
			_, predicted = prediction.max(1)
			total += batch_y.size(0)
			correct += predicted.eq(batch_y).sum().item()

	print('test acc: {}'.format(correct / total))

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
	for step, (batch_x, batch_y) in enumerate(Train_loader):  # for each training step

		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device, dtype=torch.long)

		prediction = net(batch_x)  # input x and predict based on x

		loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
		loss_train += loss.item()

		optimizer.zero_grad()  # clear gradients for next train
		loss.backward()  # backpropagation, compute gradients
		optimizer.step()  # apply gradients

		_, predicted = prediction.max(1)
		total += batch_y.size(0)
		correct += predicted.eq(batch_y).sum().item()

	loss_train = loss_train / (step + 1)
	train_acc = correct / total
	print('epoch: {}, avg_train_loss: {}, training acc:{}'.format(epoch, loss_train, train_acc))

	return net

def classifiers(Train, labelsTrain, Test, labelsTest, classifier):

	N, D = Train.shape
	if classifier == 'SVM':
		clf = linear_model.SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, alpha=0.01, random_state=0) # MNIST
		clf.fit(Train, labelsTrain)
		pred = clf.predict(Test)
		acc = metrics.accuracy_score(labelsTest, pred)
		print(acc)
		confusion = metrics.confusion_matrix(labelsTest, pred)
		print(confusion)
	elif classifier == 'LR':
		clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, alpha=0.01, random_state=0) # MNIST
		clf.fit(Train, labelsTrain)
		pred = clf.predict(Test)
		acc = metrics.accuracy_score(labelsTest, pred)
		print(acc)
		confusion = metrics.confusion_matrix(labelsTest, pred)
		print(confusion)

	return acc

def KNN_detection(Train, labelsTrain, k=10):

	# inputs: training set features and labels, k (# nearest neighbors) 
	# output: indicators of conjectured malicious samples, 0-normal, 1-malicious, it has the same size as labelsTrain

	return 

def DataPoisoning(N0, N1, N2, N3, N4):
	# Ni: # malicious samples from class i
	# cats = [0,1,2,3,4]: 5 classes of MNIST

	Attack = [N0, N1, N2, N3, N4]

	# 1.load datasets
	transform_train = transforms.Compose([
		transforms.ToTensor(),
	])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
	])
	trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_train)
	testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform_test)
	attackset = cp.copy(trainset)

	# 2. the experiments only involve 5 classes of MNIST [0, 1, 2, 3, 4]
	# split the training set, 2000 samples per class for training, 1000 samples per class for testing, 800 samples per class for poisoning
	NC = len(cats)
	nTrain = 2000
	nTest = 1000
	NAttack = 800

	# labels are stored in a list in datasets, convert it to numpy array
	trainset.targets = np.array(trainset.targets)
	attackset.targets = np.array(attackset.targets)
	testset.targets = np.array(testset.targets)
	# store indices of training samples and malicious samples
	ind_training = np.array([], dtype=int)
	ind_attack = np.array([], dtype=int)
	for c in cats:
		# randomly select NTrain+NAttack training samples from class c in [0,1,2,3,4]
		ind_c = np.where(trainset.targets == c)[0] 
		ind_c = np.random.choice(ind_c, nTrain+NAttack, False)
		# use a mask to split training samples and malicious samples
		mask = np.ones(ind_c.shape, dtype=bool)
		mask[:NAttack] = False
		# NTrain samples are used for training, NAttack samples are used for poisoning
		ind_training = np.concatenate((ind_training, ind_c[mask]))
		ind_attack = np.concatenate((ind_attack, ind_c[np.logical_not(mask)]))
	# extract samples by indices
	trainset.targets = trainset.targets[ind_training]
	trainset.data = trainset.data[ind_training]
	attackset.targets = attackset.targets[ind_attack]
	attackset.data = attackset.data[ind_attack]

	# store indices of test samples
	ind_test = np.array([], dtype=int)
	# only keep test samples from the 5 classes
	for c in cats:
		ind_c = np.where(testset.targets == c)[0]
		ind_test = np.concatenate((ind_test, ind_c))
	testset.targets = testset.targets[ind_test]
	testset.data = testset.data[ind_test]

	# groundtruth indicators of malicious samples, 0-normal, 1-malicious
	Attacks = np.zeros(trainset.targets.shape)

	# 3. poison the training set
	# Attack=[N0,N1,N2,N3,N4]
	# if Ni==1, then evenly distribute samples from class i of the attack set to the training sets of classes c!=i
	for j, attack in enumerate(Attack):
		print(cats[j], attack)
		if attack != 0:
			ind = np.where(attackset.targets == cats[j])[0]
			trainset.data = torch.cat([trainset.data, attackset.data[ind]], dim=0)
			Attacks = np.concatenate((Attacks, np.ones(attackset.targets[ind].shape)))
			label = np.array([])
			for i in cats:
				if i == cats[j]:
					continue
				label = np.concatenate((label, np.ones(int(NAttack/(NC-1)))*i))
				print('add {} labeled {}'.format(int(NAttack/(NC-1)), i))
			trainset.targets = np.concatenate((trainset.targets, label))

	return trainset, testset, Attacks

def main(N0, N1, N2, N3, N4, classifier='SVM'):
	
	# 1. poison training set
	# Attacks: groundtruth indicators of malicious samples, 0-normal, 1-malicious
	# Note: it cannot be used in anomaly detection. we only use it in performance evaluation
	trainset, testset, Attacks = DataPoisoning(N0, N1, N2, N3, N4)
	NC = len(cats)

	# flatten feature matrix, N*28*28 => N*784
	features_train = trainset.data.numpy().reshape(len(trainset.targets), -1)
	features_test = testset.data.numpy().reshape(len(testset.targets), -1)

	# 2. poisoned acc
	if classifier == 'SVM':
		acc = classifiers(features_train, trainset.targets, features_test, testset.targets, classifier)
	else:
		BATCH_SIZE = 32 
		Test_loader = Data.DataLoader(
			dataset=testset,
			batch_size=BATCH_SIZE,
			shuffle=True, num_workers=1, )
		model_path = './models/poisoned_{}_{}_{}_{}_{}.pth'.format(N0, N1, N2, N3, N4)
		net = ResNet18(num_classes=NC)
		net.to(device)
		net.load_state_dict(torch.load(model_path))
		net.eval()
		test(net, Test_loader)

	# 3. KNN-based detector
	# TODO
	# inputs: training set features and labels, k(# nearest neighbors)
	# output: A, indicators of conjectured malicious samples, 0-normal, 1-malicious, it has the same size as trainset.targets
	A = KNN_detection(features_train, trainset.targets, k=10)

	# 4. performance of the anomaly detector
	# 1)True positive rate(tpr): # truly detected malicious samples/ total # malicious samples
	# 2)False positive rate(fpr): # falsely detected malicious samples/ total # nomal samples
	# 3)acc of the classifier trained on the sanitized dataset(with detected malicious samples removed)
	TrueAttack = A*Attacks
	FalseAttack = A*np.logical_not(Attacks)
	if N0 + N1 + N2 + N3 + N4 != 0:
		tpr = TrueAttack.sum()/(Attacks.sum())
	else:
		tpr = 0
	fpr = FalseAttack.sum()/((1-Attacks).sum())
	print("tpr: {}, fpr: {}".format(tpr, fpr))

	if classifier == 'SVM':
		acc = classifiers(features_train[np.logical_not(A)], trainset.targets[np.logical_not(A)], features_test, testset.targets, classifier)
	else:
		trainset.data = trainset.data[np.logical_not(A)]
		trainset.targets = trainset.targets[np.logical_not(A)]
		Train_loader = Data.DataLoader(
			dataset=trainset,
			batch_size=BATCH_SIZE,
			shuffle=True, num_workers=1, )
		EPOCH = 30
		model_path = './models/KNN_sanitized_{}_{}_{}_{}_{}.pth'.format(N0, N1, N2, N3, N4)
		net = ResNet18(num_classes=NC)
		net.to(device)
		for epoch in range(EPOCH):
			net = train(epoch, net, Train_loader)
			test(net, Test_loader)
		torch.save(net.state_dict(), model_path)

if __name__ == "__main__":
	main(N0=1, N1=1, N2=1, N3=1, N4=0, classifier='SVM')