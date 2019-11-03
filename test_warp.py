import numpy as np
from mtcnn.mtcnn import MTCNN
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from scipy.ndimage import map_coordinates


detector = MTCNN()

img = io.imread('img.jpg')
print('test', img[0][0])
print('test', img.dtype)

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
print("rect", rect)


"""
sampleX = np.expand_dims(np.arctanh(np.linspace(-.99, .99, 512)), 0).repeat(512, 0)
sampleY = np.expand_dims(np.arctanh(np.linspace(-.99, .99, 512)),-1).repeat(512,-1)
coords = np.stack([sampleX, sampleY]).transpose(1,2,0)
coords = np.flip(tform.inverse(coords.reshape(-1,2)).reshape(512,512,2).transpose(2,0,1), 0)

warped0 = map_coordinates(img[:,:,0], coords)
warped1 = map_coordinates(img[:,:,1], coords)
warped2 = map_coordinates(img[:,:,2], coords)
warped = np.stack([warped0, warped1, warped2]).transpose(1,2,0)
"""

def map_func1(coords):
	tform2 = transform.SimilarityTransform(scale=1./257., rotation=0, translation=(-0.99, -0.99))
	return tform.inverse(np.arctanh(tform2(coords)))

def map_func2(coords):
	tform2 = transform.SimilarityTransform(scale=257., rotation=0, translation=(255.5, 255.5))
	return tform2(np.tanh(tform(coords)))


warped = transform.warp(img, inverse_map=map_func1, output_shape=[512,512] )
warped_inv = transform.warp(warped, inverse_map=map_func2, output_shape=img.shape )


print('test', warped[0][0])
print('test', warped_inv[0][0])
print('test', warped.max(), warped_inv.max())
print('test', warped.dtype, warped_inv.dtype)


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
plt.close()

fig,ax = plt.subplots(1)
ax.imshow(warped_inv)
plt.show()