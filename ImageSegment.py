import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans

img = cv2.imread("Airplane.jpg")  # or "Tiger.jpg"
bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# read image by cv2
# cv2.imshow("before", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# read image by matplot. Before clustering
plt.figure("before")
plt.axis("off")
plt.imshow(bgr2rgb)  # needs to be converted from bgr to rgb
plt.show()

# normalize and reshape image into a 1D array
img = bgr2rgb / 255
img_1d = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

# apply K-means Clustering. Values can be changed
kmean = KMeans(n_clusters=5, random_state=0).fit(img_1d)

# reconstruct image with clustered centeroids
new_img = kmean.cluster_centers_[kmean.labels_]

# reshape the image back to its original shape
clusteredimg = new_img.reshape(img.shape[0], img.shape[1], img.shape[2])

# display image after clustering
plt.figure("after")
plt.axis("off")
plt.imshow(clusteredimg)
plt.show()
