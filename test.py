import numpy as np
from mtcnn.mtcnn import MTCNN
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

detector = MTCNN()

img = io.imread('img.jpg')
res = detector.detect_faces(img)[0]['keypoints']
rows, cols = [], []

for v in res.values():
	rows.append(v[0])
	cols.append(v[1])

# A*P = R
# A = R*P_inv

P = np.array([res['left_eye'], res['mouth_right']]).T
R = np.array([[-0.25, -0.1], [0.15, 0.4]]).T
A = R.dot(np.linalg.inv(P))
A_inv = np.linalg.inv(A)

print('A', A)
print('A_inv', A_inv)
rect = np.array([[-1,1], [1,1], [1,-1], [-1,-1]]).T
print('r_before', rect)
rect = A_inv.dot(rect).T
print('r_after', rect)

fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
rect = patches.Polygon(rect, linewidth=2,edgecolor='g', fill=False)

# Add the patch to the Axes
ax.add_patch(rect)

plt.scatter(rows,cols, s=10,marker='x',c='r')
plt.show()