import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def epipolar_error(pt1, pt2, F, img1=None, img2=None):
    """
    Symmetric epipolar distance for points pt1 (cam1), pt2 (cam2),
    with visualization of epipolar lines and points.
    """
    pt1_h = np.array([pt1[0], pt1[1], 1.0])
    pt2_h = np.array([pt2[0], pt2[1], 1.0])

    l2 = F @ pt1_h  # Epipolar line in cam2
    l1 = F.T @ pt2_h # Epipolar line in cam1

    # Normalize the lines
    norm_l1 = np.linalg.norm(l1[:2])
    norm_l2 = np.linalg.norm(l2[:2])

    if norm_l1 > 1e-8:
        l1 /= norm_l1
    if norm_l2 > 1e-8:
        l2 /= norm_l2

    d1 = abs(np.dot(l1, pt1_h)) if norm_l1 > 1e-8 else 9999
    d2 = abs(np.dot(l2, pt2_h)) if norm_l2 > 1e-8 else 9999

    error = 0.5 * (d1 + d2)

    # Visualization
    if img1 is not None and img2 is not None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Calculate line endpoints for plotting (more robust)
        def line_end_points(line, h, w):
          if abs(line[0]) > abs(line[1]):
            x0 = -line[2] / line[0]
            x1 = -(line[2] + line[1]*h) / line[0]
            y0 = 0
            y1 = h
          else:
            y0 = -line[2] / line[1]
            y1 = -(line[2] + line[0]*w) / line[1]
            x0 = 0
            x1 = w
          return (int(x0), int(y0)), (int(x1), int(y1))
        
        p1_1, p1_2 = line_end_points(l1, h1, w1)
        p2_1, p2_2 = line_end_points(l2, h2, w2)

        # Plotting using Matplotlib
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1 is not None else np.zeros((h1,w1,3)), cmap='gray') # Handle cases where the images may not be provided
        plt.plot(pt1[0], pt1[1], 'ro', label='pt1')  # Red dot for pt1
        plt.plot([p1_1[0], p1_2[0]], [p1_1[1], p1_2[1]], 'g-', label='Epipolar Line 1') # Green line for l1
        plt.title('Image 1')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2 is not None else np.zeros((h2,w2,3)), cmap='gray')
        plt.plot(pt2[0], pt2[1], 'bo', label='pt2')  # Blue dot for pt2
        plt.plot([p2_1[0], p2_2[0]], [p2_1[1], p2_2[1]], 'm-', label='Epipolar Line 2') # Magenta line for l2
        plt.title('Image 2')
        plt.legend()

        plt.show()

    return error

def epipolar_error_full(pt1, pt2, pt3, F12, F13, F23):
    """
    epipolar error across three cams:
      e12 + e13 + e23
    """
    e12 = epipolar_error(pt1, pt2, F12)
    e13 = epipolar_error(pt1, pt3, F13)
    e23 = epipolar_error(pt2, pt3, F23)
    return (e12 + e13 + e23) / 3

def compute_cost_matrix(dets1, dets2, dets3, F12, F13, F23, img1=None, img2=None, img3=None):
    """
    NxMxP cost matrix from bounding-box centers.
    """
    N, M, P = len(dets1), len(dets2), len(dets3)
    cost = np.zeros((N, M, P), dtype=np.float32)

    for i in range(N):
        pt1 = dets1[i]['bb_center']
        for j in range(M):
            pt2 = dets2[j]['bb_center']
            for k in range(P):
                pt3 = dets3[k]['bb_center']
                cost[i,j,k] = epipolar_error_full(pt1, pt2, pt3, F12, F13, F23)

    return cost

def match_objects(cost_matrix, threshold):
    """
    Flatten => Hungarian => keep matches < threshold => list of (i, j, k).
    """
    N, M, P = cost_matrix.shape
    matched = []
    flattened = cost_matrix.reshape(N*M, P)
    row_idx, col_idx = linear_sum_assignment(flattened)

    for r, c in zip(row_idx, col_idx):
        val = flattened[r, c]
        if val < threshold:
            i = r // M
            j = r % M
            k = c
            matched.append((i, j, k))
    return matched

def triangulate_multi_view(proj_mats, points_2D):
    """Triangulate using Direct Linear Transform."""
    A = []
    for P, (x, y) in zip(proj_mats, points_2D):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]  # Non-homogeneous coordinates