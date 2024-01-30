import numpy as np
# import pickle
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
	# test acc
	total = 0
	correct = 0
	with torch.no_grad():
		for step, (batch_x, batch_y) in enumerate(Test_loader):  # for each training step

			# TODO: 
			# feed batch_x into the neural network
			# compute test accuracy based on the output and batch_y

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

		# TODO: 
		# feed batch_x into the neural network
		# compute loss, update parameters of the neural network
		# compute training accuracy based on the output and batch_y

	loss_train = loss_train / (step + 1)
	train_acc = correct / total
	print('epoch: {}, avg_train_loss: {}, training acc:{}'.format(epoch, loss_train, train_acc))

	return net


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
	cats = [0, 1, 2, 3, 4]
	NC = len(cats)
	nTrain = 2000
	nTest = 1000
	NAttack = 800

	# labels are stored in a list in datasets, convert to numpy array
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

	# 3. poison the training set
	# Attack=[N0,N1,N2,N3,N4]
	# if Ni==1, then evenly distribute samples from class i of the attack set to the training sets of classes c!=i
	for j, attack in enumerate(Attack):
		if attack != 0:
			# TODO:
			# select samples of cats[j] from the attackset
			# put those samples into the training set
			# mislabel those samples as c != cats[j], that is,
			# 	for each class c!=cats[j], put NAttack/(NC-1) labels of c in trainset.targets

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
	if not os.path.isdir('./models/'):
		os.makedirs('./models/')
	model_path = './models/poisoned_{}_{}_{}_{}_{}.pth'.format(N0, N1, N2, N3, N4)
	torch.save(net.state_dict(), model_path)

if __name__ == "__main__":
	DataPoisoning(N0=0, N1=0, N2=0, N3=0, N4=0)