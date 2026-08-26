This repository contains a structured Python solution to perform image stitching using traditional computer vision techniques. The pipeline processes three or more overlapping images to automatically construct a unified panoramic canvas.

---

## 📋 Exam Requirements & Implementation Pipeline

The exam is executed in 7 consecutive steps, outlined below with their technical mechanism:

### 1. Detect SIFT Features
* **Mechanism:** The Scale-Invariant Feature Transform (SIFT) algorithm applies a Difference-of-Gaussians (DoG) filter across multiple scale spaces to identify highly stable local extrema (points of interest such as corners, edges, and high-contrast vertices). These keypoints are selected because they are uniquely identifiable and invariant to scale and rotation.
Our implementation detected 35847 keypoints in left image and 27252 in right image.

### 2. Extract Descriptors
* **Mechanism:** For every detected keypoint, SIFT computes the gradient magnitude and orientation within its local neighborhood. This localized gradient distribution is mapped into an 8-bin orientation histogram across a 4x4 grid, culminating in a highly distinctive **128-dimensional descriptor vector** for each keypoint. These vectors remain robust against changes in illumination, noise, and 3D viewpoint alterations.


### 3. Perform Matching
* **Mechanism:** To correlate keypoints between the two images, we compute the Euclidean distance ($L_2$ norm) across their respective 128-dimensional descriptor spaces. A Brute-Force Matcher combined with $k$-Nearest Neighbors ($k$-NN where $k=2$) fetches the two closest matching features in the target image for each feature in the source image.

### 4. Apply Lowe's Ratio Test
* **Mechanism:** To filter out false matches resulting from structural noise or ambiguous background patterns, Lowe's Ratio Test evaluates the uniqueness of a match. For a given keypoint, if the distance to the closest match ($m$) is not significantly smaller than the distance to the second-closest match ($n$), the match is rejected as ambiguous. The standard filtering threshold is defined as:
  $$\\frac{\\text{Distance}(m)}{\\text{Distance}(n)} < 0.75$$
  The number of dood matches after Lowe's Ratio Test is 1416
  The file "matched.jpg" displays an image of matching features in left and right images.

### 5. Estimate Homography using RANSAC
* **Mechanism:** A Homography matrix ($H$) is a $3 \\times 3$ projective transformation matrix that maps coordinates from one 2D image plane to another ($x_2 = H x_1$). Because the initial match set always contains false positives (outliers), **RANSAC (RANdom Sample Consensus)** is employed. It iteratively:
  1. Selects 4 random matching pairs (the minimum required to solve for 8 degrees of freedom).
  2. Computes a trial homography matrix.
  3. Validates the matrix against all remaining matches, tallying the number of "inliers" falling within a geometric distance threshold. For instance, 5 pixels.
  The homography matrix with the maximum number of consensus inliers is selected as the optimal model.

### 6. Produce a Panorama
* **Mechanism:** The source image is transformed into the coordinate space of the destination canvas using `cv2.warpPerspective` with the calculated homography matrix $H$. The second image is then laid down as the anchor frame. Finally, the canvas is automatically cropped to remove extraneous black borders generated during the geometric warping process.
The Panorama image generated is save as final_panorama_[Feature Detector].jpg

### 7. Analyze Failure Cases
Even robust SIFT/ORB + RANSAC frameworks fail under specific environmental and geometric bounds:
* **Textureless / Low-Contrast Environments:** Scenes dominated by featureless surfaces such as blank walls, smooth skies, and heavy fog do not yield sufficient local gradients. SIFT fails to extract keypoints, preventing homography computation entirely.
* **Repetitive Patterns (Perceptual Ambiguity):** Structures like building facades with identical window grids, fences, or tiling cause multiple keypoints to yield nearly identical descriptor vectors. Consequently, Lowe's Ratio Test filters them out as ambiguous, leaving insufficient data for RANSAC.
* **Geometric Parallax (Translational Camera Movements):** Pure homography strictly assumes either a completely flat 2D scene or that the camera underwent pure rotation around its optical center. If the camera physically translates between shots, objects at varying depths shift relative to each other (parallax), creating severe blending ghosts and stitch misalignments.
* **Dynamic / Moving Elements:** If elements change position within the overlapping zone between shots such as moving pedestrians or cars, the system will either generate dual artifacts (ghosts), cut the objects in half, or discard the regions as outliers during RANSAC.


### 8. Results
=== RUNNING ORB PIPELINE ===
[✓] Saved prepared grayscale image: prepared_grayscale_img_1.jpg
[✓] Saved prepared grayscale image: prepared_grayscale_img_2.jpg
[✓] Saved prepared grayscale image: prepared_grayscale_img_3.jpg
[✓] Saved keypoints visualization (3000 points): keypoints_img_1_ORB.jpg
[✓] Saved keypoints visualization (3000 points): keypoints_img_2_ORB.jpg
[✓] Saved keypoints visualization (3000 points): keypoints_img_3_ORB.jpg

--- Metrics for Pair 1 [ORB] ---
Estimated Homography Matrix:
[[ 1.00271947e+00 -3.02590422e-04 -6.35548616e+02]
 [ 5.52993816e-03  9.99713846e-01 -1.26054884e+02]
 [ 4.62011246e-06 -2.51281671e-06  1.00000000e+00]]
Keypoints in Source: 3000 | Keypoints in Destination: 3000
Initial Matches (Lowe's Test): 457
RANSAC Inliers: 429
Inlier Ratio: 93.87%
Extraction & Matching Time: 0.0362s

--- Metrics for Pair 2 [ORB] ---
Estimated Homography Matrix:
[[ 9.71387623e-01 -3.46792910e-03 -8.30850644e+02]
 [-2.30065927e-02  9.72822566e-01 -9.41904929e+01]
 [-2.53958388e-05 -2.00553330e-06  1.00000000e+00]]
Keypoints in Source: 3000 | Keypoints in Destination: 3000
Initial Matches (Lowe's Test): 267
RANSAC Inliers: 204
Inlier Ratio: 76.40%
Extraction & Matching Time: 0.0369s

[✓] Finished ORB Panorama in 0.3804s total.
[✓] Saved final panorama: final_panorama_ORB.jpg
---

## 💻 Set up environment
```bash
python3 -m venv venv 
source venv/bin/activate
```

## 💻 Prerequisites & Setup

Ensure you have Python installed alongside the following dependencies:

```bash
pip install opencv-python numpy matplotlib
python 22424761.py
```

## Transforming View
```bash
  python transformimage.py
```

Adjust ``` test_image_paths = ['testImages/left.png', 'testImages/right.png', 'testImages/top.png'] ``` in 22424761.py with route for images you want to use as test data
