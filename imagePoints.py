import cv2
import numpy as np

# Assuming you have an image already
image = cv2.imread('test.jpg')

# Assume points1 and points2 are numpy arrays of size (100, 2)
points1 = np.random.randint(0, high=image.shape[0], size=(100, 2))
points2 = np.random.randint(0, high=image.shape[0], size=(100, 2))

# Draw points on the image
for pt1, pt2 in zip(points1, points2):
    image = cv2.circle(image, tuple(pt1), radius=5, color=(0, 255, 0), thickness=-1)
    image = cv2.circle(image, tuple(pt2), radius=5, color=(0, 0, 255), thickness=-1)
    image = cv2.line(image, tuple(pt1), tuple(pt2), color=(255, 0, 0), thickness=2)

cv2.imwrite("testout.jpg",image)