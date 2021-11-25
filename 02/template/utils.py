import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    # 
    # Output: T 3x3 transformation matrix of points

    # TODO TASK:
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid
    centroid = np.mean(x, 1)
    centered_image = np.apply_along_axis(lambda point: point - centroid, 0, x)
    distance = np.array([])
    # print(np.mean(centered_image, 1))
    for i in range(centered_image.shape[1]):
        temp = np.zeros(3)
        temp[0] = centered_image[0, i]
        temp[1] = centered_image[1, i]
        temp[2] = centered_image[2, i]
        distance = np.append(distance, np.linalg.norm(temp))

    mean_distance = np.mean(distance)
    scale_factor = np.sqrt(2)/mean_distance
    # print(mean_distance)
    # print(scale_factor)
    # print(1/mean_distance)

    return np.array([[scale_factor, 0, -centroid[0] * scale_factor],
                     [0, scale_factor, -centroid[1] * scale_factor],
                     [0, 0, 1]])


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        T1 = get_normalization_matrix(x1)
        T2 = get_normalization_matrix(x2)

        # Normalize inputs
        n1 = np.apply_along_axis(lambda point: np.matmul(T1, point), 0, x1)
        n2 = np.apply_along_axis(lambda point: np.matmul(T2, point), 0, x2)

    # Construct matrix A encoding the constraints on x1 and x2
    A = np.zeros((N, 9))
    for i in range(N):
        p1 = n1[:, i]
        p2 = n2[:, i]
        A[i] = np.array([p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1] * p1[1], p2[1], p1[0], p1[1], 1])

    # Solve for f using SVD
    u, s, v = np.linalg.svd(A)
    F = v.T[:, 8].reshape((3, 3))

    # Enforce that rank(F)=2
    u, s, v = np.linalg.svd(F)
    s_next = np.array([[s[0], 0, 0],
                       [0, s[1], 0],
                       [0, 0, 0]])
    F = np.matmul(u, s_next)
    F = np.matmul(F, v)

    if normalize:
        F = np.matmul(np.matmul(T2.T, F), T1)

    return F


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)

    The epipole is the point, where the camera, that generates the other image, would be rendered
    """

    # The epipole is the null space of F (F * e = 0)
    e = null_space(F)
    return e/e[2]


def plot_epipolar_line(im, F, x, e, plot=plt):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.

    The epipolar line corresponding to x is the line on which x could be in the other image.
    It is aswell the intersection between image plane and epipolar plane
    """
    m, n = im.shape[:2]
    epipolar_line = np.dot(F, x)
    x = np.array(range(n))
    y = np.apply_along_axis(lambda entry: (epipolar_line[0] * entry + epipolar_line[2]) / (- epipolar_line[1]), 0, x)
    valid_range = (y >= 0) & (y < m)
    plot.plot(x[valid_range], y[valid_range], linewidth=2)
    # plot.plot(e[0], e[1], 'ro')
