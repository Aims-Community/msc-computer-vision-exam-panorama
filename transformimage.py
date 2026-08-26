import cv2
import numpy as np

base_img = cv2.imread('testImages/left.png')
h, w = base_img.shape[:2]
center = (w // 2, h // 2)

# Rotation (30 degrees counter-clockwise)
M_rot = cv2.getRotationMatrix2D(center, 30, 1.0)
img_rot = cv2.warpAffine(base_img, M_rot, (w, h))
cv2.imwrite('testImages/Left_rotation.png', img_rot)

# Scale Change to 1.5x zoom
img_scaled = cv2.resize(base_img, (int(w * 1.5), int(h * 1.5)))
# Crop back to canvas size to simulate a zoomed-in camera shot
img_scaled = img_scaled[int(h*0.25):int(h*1.25), int(w*0.25):int(w*1.25)]
cv2.imwrite('testImages/Left_scale.png', img_scaled)

# Viewpoint (35-degree oblique perspective tilt)
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_pts = np.float32([[w*0.1, h*0.05], [w*0.85, 0], [w, h*0.95], [w*0.05, h*0.85]])
M_view = cv2.getPerspectiveTransform(src_pts, dst_pts)
img_view = cv2.warpPerspective(base_img, M_view, (w, h))
cv2.imwrite('testImages/Left_viewpoint.png', img_view)

# Illumination Change (simulate heavy shadow / low exposure)
img_illum = cv2.convertScaleAbs(base_img, alpha=0.5, beta=-30)
cv2.imwrite('testImages/Left_illumination.png', img_illum)