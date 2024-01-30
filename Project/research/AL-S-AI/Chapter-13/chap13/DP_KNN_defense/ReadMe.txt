Defense against Data poisoning 
Goal: Detect poisoned images in MNIST
Method: Relabel an image as the plurality label of its K nearest neighbors.
Reference: Label sanitization against label flipping poisoning attacks, https://arxiv.org/abs/1803.00992  This paper provides a KNN based anomaly detector for binary classification tasks. It's trivial to extend it for multi-class classification tasks.
Experiments:
1. Poison the training set of MNIST, keep the groundtruth indicators of malicious samples, 0-clean, 1-malicious. (Note: The groudtruth indicators are used in performance evaluation in step 4, and they are NOT allowed in anomaly detection.)
2. Train a SVM classifier on the poisoned training set or load the resnet-18 trained in the last project. (For those who are using a CPU, training a DNN is expensive. You can first do the experiments on a SVM classifier. Also, to load your own resnet-18, just modify the model path.)
3. (TODO) Implement the KNN based defense with k a hyper-parameter(# nearest neighbors). The default value of k is 10. 
	For each training sample:
	1) Find its k nearest neighbors. (Check sklearn.neighbors for functions of finding k nearest neighbors of a data sample.)
	2) Find the plurality label of the k nearest neighbors. (Check scipy.stats for functions of finding the plurality.)
	3) If the plurality label is different from its original label, it is deemed a malicious sample. Otherwise, it is clean.
4. Evaluate the performance of the KNN based detector by:
	1) True positive rate(TPR): # truly detected malicious samples/ total # malicious samples
	2) False positive rate(FPR): # falsely detected malicious samples/ total # clean samples
	3) ACC of the classifier trained on sanitized dataset (with the detected malicious samples removed)
5. Vary the value of k, observe how it affects the performance of the detector. (For those who are using a CPU, you can observe the effects of k on a SVM.)

