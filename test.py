import numpy as np
from mtcnn.mtcnn import MTCNN
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from scipy.ndimage import map_coordinates


detector = MTCNN()

img = io.imread('img.jpg')
h,w,c = img.shape
res = detector.detect_faces(img)[0]['keypoints']
rows, cols = [], []

for v in res.values():
	rows.append(v[0])
	cols.append(v[1])

src = np.array([res['left_eye'], res['right_eye'], res['nose'], res['mouth_left'], res['mouth_right']])
dst = np.array([[-0.25,-0.1], [0.25, -0.1], [0.0, 0.1], [-0.15, 0.4], [0.15, 0.4]])
tform = transform.estimate_transform('similarity', src, dst)

rec = [[-1.,1.], [1.,1.], [1.,-1.], [-1.,-1.]]
rect = tform.inverse(rec)

sampleX = np.expand_dims(np.arctanh(np.linspace(-.99, .99, 512)),-1).repeat(512,-1)
sampleY = np.expand_dims(np.arctanh(np.linspace(-.99, .99, 512)),-1).repeat(512,-1)

coords = np.stack([sampleY, sampleX]).transpose(1,2,0)
print(coords.reshape(-1,2)[512*512//2:512*512//2+10])


coords = tform.inverse(coords.reshape(-1,2)).reshape(512,512,2).transpose(2,0,1)
print(coords[0,256:266,256:266])

warped0 = map_coordinates(img[:,:,0], coords)
warped1 = map_coordinates(img[:,:,1], coords)
warped2 = map_coordinates(img[:,:,2], coords)
warped = np.stack([warped0, warped1, warped2]).transpose(1,2,0)

print('img', img.shape)
print('warp', warped.shape)

print("rect", rect)



fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
rect = patches.Polygon(rect, linewidth=2,edgecolor='g', fill=False)

# Add the patch to the Axes
ax.add_patch(rect)

plt.scatter(rows,cols, s=10,marker='x',c='r')
plt.show()

plt.close()

fig,ax = plt.subplots(1)
ax.imshow(warped)
plt.show()

