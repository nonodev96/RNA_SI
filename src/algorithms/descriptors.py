import cv2


def describe_image_with_sift(image):
    sift = cv2.SIFT.create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
