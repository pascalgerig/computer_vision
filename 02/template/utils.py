import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import lsmr


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
    for i in range(centered_image.shape[1]):
        temp = np.zeros(3)
        temp[0] = centered_image[0, i]
        temp[1] = centered_image[1, i]
        temp[2] = centered_image[2, i]
        distance = np.append(distance, np.linalg.norm(temp))

    mean_distance = np.mean(distance)
    scale_factor = np.sqrt(2)/mean_distance

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
    else:
        n1 = x1
        n2 = x2

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


def ransac(x1, x2, threshold, num_steps=1000, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible
    best_inliers_count = 0
    best_inliers = None

    for _ in range(num_steps):
        # Randomly sample 8 point corresponences
        selectedPointIndizes = np.random.choice(x1.shape[1], 8, replace=False)
        im1 = np.array(x1[:, selectedPointIndizes])
        im2 = np.array(x2[:, selectedPointIndizes])

        # Calculate F from 8 point correspondences
        temp_F = eight_points_algorithm(im1, im2)

        error_indicator = np.square(np.sum(x2 * (np.matmul(temp_F, x1)), axis=0))
        temp_inliers = error_indicator < threshold
        # TODO double check this with mean()
        temp_inliers_count = temp_inliers.sum()

        # F is currently best if number of inliers > previous highest count of inliers
        if temp_inliers_count > best_inliers_count:
            best_inliers_count = temp_inliers_count
            best_inliers = temp_inliers

    while True:
        F = eight_points_algorithm(x1[:, best_inliers], x2[:, best_inliers])
        error_indicator = np.square(np.sum(x2 * (np.matmul(F, x1)), axis=0))
        inliers = error_indicator < threshold
        if (inliers == best_inliers).all():
            return F, inliers  # F is estimated fundamental matrix and inliers is an indicator (boolean) numpy array
        best_inliers = inliers


def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # TODO: Compute possible rotations and translations
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V) < 0:
        V *= -1

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    R1 = U @ W.T @ V
    R2 = U @ W @ V
    t1 = U[:,2].reshape(3, 1)
    t2 = -t1

    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d
