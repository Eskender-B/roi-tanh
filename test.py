import numpy as np
from mtcnn.mtcnn import MTCNN
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy.core.symbol import symbols
from sympy.solvers.solvers import nsolve
from sympy.solvers.solveset import nonlinsolve
from sympy import cos
import math

detector = MTCNN()

img = io.imread('img.jpg')
h,w,c = img.shape
res = detector.detect_faces(img)[0]['keypoints']
rows, cols = [], []

for v in res.values():
	rows.append(v[0])
	cols.append(v[1])

"""
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
"""

x0, y0, s, cos_th = symbols('x0, y0, s, cos_th', real=True)
"""
eqs = [-0.25/s - (float(res['left_eye'][0])-x0)/ (1.-th**2/4),
		#0.25/s - (float(res['right_eye'][0])-x0)/cos(th),
		#0.0/s - (float(res['nose'][0])-x0)/cos(th),
		#-0.15/s - (float(res['mouth_left'][0])-x0)/cos(th),
		0.15/s - (float(res['mouth_right'][0])-x0)/(1.-th**2/4),

		-0.1/s - (float(res['left_eye'][1])-y0)*(1.-th**2/4),
		#-0.1/s - (float(res['right_eye'][1])-y0)*cos(th),
		#0.1/s - (float(res['nose'][1])-y0)*cos(th),
		#0.4/s - (float(res['mouth_left'][1])-y0)*cos(th),
		0.4/s - (float(res['mouth_right'][1])-y0)*(1.-th**2/4)
	  ]
"""

eqs = [
		-0.25*-0.1 - (float(res['left_eye'][0])-x0)*(float(res['left_eye'][1])-y0)*s**2,
		0.25*-0.1 - (float(res['right_eye'][0])-x0)*(float(res['right_eye'][1])-y0)*s**2,
		0.0*0.1 - (float(res['nose'][0])-x0)*(float(res['nose'][1])-y0)*s**2,
		-0.1 - cos_th*(s*(float(res['right_eye'][1])-y0))
		]
ans = nsolve(eqs, [x0,y0,s, cos_th], [w/2., h/2., 1/128., 1.])
ans = ans.tolist()


print("init", w/2.,h/2.,1/128., 1.)
print("ans", ans)

x0, y0, s, cos_th = ans[0][0], ans[1][0], ans[2][0], ans[3][0]
th = math.degrees(math.acos(ans[3][0]))
print("angle", th)

def transform(pt):
	xp,yp = pt
	x = x0 + xp*cos_th/s
	y = y0 + yp/(cos_th*s)
	return [x, y]

rec = [[-1.,1.], [1.,1.], [1.,-1.], [-1.,-1.]]
rect = []
for r in rec:
	rect.append(transform(r))

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