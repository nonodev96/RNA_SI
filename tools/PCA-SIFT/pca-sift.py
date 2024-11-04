import cv2
import numpy as np

# Load fingerprint image
fingerprint_image = cv2.imread('/home/nonodev96/Projects/RNA_SI/datasets/ruizgara/socofing/versions/2/SOCOFing/Real/1__M_Left_index_finger.BMP', 0)

# Create SIFT object
sift = cv2.SIFT.create()

# Detect keypoints and compute SIFT descriptors
keypoints, descriptors = sift.detectAndCompute(fingerprint_image, None)

# Use PCA to reduce dimensionality of SIFT descriptors
num_components = 128  # Number of principal components to retain
mean = np.empty((0))
mean, eigenvectors, eigenvalues = cv2.PCACompute2(descriptors, mean)
print(mean, eigenvectors, eigenvalues)

# pca_descriptors = pca.fit_transform(descriptors)

# Normalize the PCA-SIFT descriptors
normalized_pca_descriptors = cv2.normalize(pca_descriptors, None)

# Compute RootSIFT descriptors
root_sift_descriptors = np.sqrt(normalized_pca_descriptors)

# Perform L2 normalization on RootSIFT descriptors
root_sift_descriptors /= np.linalg.norm(root_sift_descriptors, axis=1).reshape(-1, 1)

# Extract the keypoints and descriptors for further use in fingerprint recognition system
fingerprint_keypoints = keypoints
fingerprint_descriptors = root_sift_descriptors

# You can now use the fingerprint_keypoints and fingerprint_descriptors in your fingerprint recognition system for matching, classification, or other tasks.