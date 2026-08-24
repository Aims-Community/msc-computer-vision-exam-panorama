import cv2
import numpy as np
import time
import os

def create_panorama(image_paths, method='SIFT', ratio_thresh=0.75, ransac_thresh=5.0, save_visualizations=True):
    """
    Creates an automatic panorama from 3+ overlapping images with full intermediate visualizations.
    
    Parameters:
    - image_paths: List of file paths to overlapping images.
    - method: 'SIFT' or 'ORB'.
    - ratio_thresh: Lowe's ratio test threshold.
    - ransac_thresh: Maximum reprojection error threshold for RANSAC.
    - save_visualizations: Saves grayscale, keypoint, match, and panorama images to disk.
    """
    start_total_time = time.time()
    method = method.upper()

    # --- 1. Image Acquisition & Grayscale Preparation ---
    images = []
    gray_images = []
    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        gray_images.append(gray)

        # Save and visualize prepared grayscale image
        if save_visualizations:
            gray_filename = f"prepared_grayscale_img_{idx + 1}.jpg"
            cv2.imwrite(gray_filename, gray)
            print(f"[✓] Saved prepared grayscale image: {gray_filename}")

    if len(images) < 2:
        raise ValueError("At least two overlapping images are required.")

    # --- 2. Initialize Feature Detector & Descriptor ---
    if method == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=3000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError("Unsupported method! Choose 'SIFT' or 'ORB'.")

    # --- 3. Keypoint Detection & Visualizations on Grayscale ---
    keypoints_list = []
    descriptors_list = []

    for idx, gray in enumerate(gray_images):
        kp, des = detector.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)

        if save_visualizations:
            # Draw detected keypoints on top of the grayscale image
            img_kp = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            kp_filename = f"keypoints_img_{idx + 1}_{method}.jpg"
            cv2.imwrite(kp_filename, img_kp)
            print(f"[✓] Saved keypoints visualization ({len(kp)} points): {kp_filename}")

    # --- 4. Inner Pair Stitching Function ---
    def stitch_pair(img_src, img_dst, pair_index=0):
        gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        kp_src, des_src = detector.detectAndCompute(gray_src, None)
        kp_dst, des_dst = detector.detectAndCompute(gray_dst, None)
        feature_time = time.time() - t0

        if des_src is None or des_dst is None or len(des_src) < 2 or len(des_dst) < 2:
            print(f"[!] Warning: Insufficient descriptors for pair {pair_index}.")
            return img_dst

        # 5. Matching & Lowe's Ratio Test
        raw_matches = matcher.knnMatch(des_src, des_dst, k=2)
        initial_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    initial_matches.append(m)

        if len(initial_matches) < 4:
            print(f"[!] Warning: Insufficient matches ({len(initial_matches)}) for pair {pair_index}.")
            return img_dst

        # 6. RANSAC Homography Estimation
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in initial_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in initial_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        inliers_count = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = (inliers_count / len(initial_matches)) * 100 if initial_matches else 0.0

        # Print Metrics
        print(f"\n--- Metrics for Pair {pair_index} [{method}] ---")
        print(f"Keypoints in Source: {len(kp_src)} | Keypoints in Destination: {len(kp_dst)}")
        print(f"Initial Matches (Lowe's Test): {len(initial_matches)}")
        print(f"RANSAC Inliers: {inliers_count}")
        print(f"Inlier Ratio: {inlier_ratio:.2f}%")
        print(f"Extraction & Matching Time: {feature_time:.4f}s")

        # 7. Visualization of Correspondences Before and After RANSAC
        if save_visualizations and mask is not None:
            # Before RANSAC
            img_before = cv2.drawMatches(img_src, kp_src, img_dst, kp_dst, initial_matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"matches_before_ransac_pair{pair_index}_{method}.jpg", img_before)

            # After RANSAC (Inliers only)
            inlier_matches = [initial_matches[i] for i in range(len(initial_matches)) if mask.ravel()[i] == 1]
            img_after = cv2.drawMatches(img_src, kp_src, img_dst, kp_dst, inlier_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"matches_after_ransac_pair{pair_index}_{method}.jpg", img_after)

        if H is None:
            return img_dst

        # 8. Geometric Transformation and Stitching
        h_src, w_src = img_src.shape[:2]
        h_dst, w_dst = img_dst.shape[:2]

        pts_src = np.float32([[0, 0], [0, h_src], [w_src, h_src], [w_src, 0]]).reshape(-1, 1, 2)
        pts_src_warped = cv2.perspectiveTransform(pts_src, H)
        pts_dst = np.float32([[0, 0], [0, h_dst], [w_dst, h_dst], [w_dst, 0]]).reshape(-1, 1, 2)

        all_pts = np.concatenate((pts_dst, pts_src_warped), axis=0)
        [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        warped_src = cv2.warpPerspective(img_src, translation.dot(H), (x_max - x_min, y_max - y_min))

        canvas = warped_src.copy()
        canvas[-y_min:h_dst - y_min, -x_min:w_dst - x_min] = img_dst
        return canvas

    # --- 9. Multi-Image Stitching Execution (Outward from Anchor) ---
    mid_idx = len(images) // 2
    panorama_result = images[mid_idx]

    pair_counter = 1
    # Stitch left side
    for i in range(mid_idx - 1, -1, -1):
        panorama_result = stitch_pair(images[i], panorama_result, pair_index=pair_counter)
        pair_counter += 1

    # Stitch right side
    for i in range(mid_idx + 1, len(images)):
        panorama_result = stitch_pair(images[i], panorama_result, pair_index=pair_counter)
        pair_counter += 1

    total_time = time.time() - start_total_time
    print(f"\n[✓] Finished {method} Panorama in {total_time:.4f}s total.")
    
    if save_visualizations:
        cv2.imwrite(f"final_panorama_{method}.jpg", panorama_result)
        print(f"[✓] Saved final panorama: final_panorama_{method}.jpg")
        
    return panorama_result

# ==============================================================================
# Execution & Verification
# ==============================================================================
if __name__ == '__main__':
    # Supply 3 overlapping images
    test_image_paths = ['testImages/left.png', 'testImages/right.png', 'testImages/top.png']

    # Run 1: SIFT Pipeline
    print("=== RUNNING SIFT PIPELINE ===")
    sift_panorama = create_panorama(test_image_paths, method='SIFT')

    # Run 2: ORB Pipeline for Comparative Analysis
    # print("\n=== RUNNING ORB PIPELINE ===")
    # orb_panorama = create_panorama(test_image_paths, method='ORB')