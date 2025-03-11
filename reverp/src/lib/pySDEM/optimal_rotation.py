import numpy as np

def optimal_rotation(v, landmark, target):
    """
    This code is written as a direct port from the original MATLAB implementation by by K. Kvist, Department of Materials and Production, Aalborg University, Denmark,
    If you use this code in your work, please cite for the original implementation:
    [1] Z. Lyu, L. M. Lui, and G. P. T. Choi,
        "Spherical Density-Equalizing Map for Genus-0 Closed Surfaces."
        SIAM Journal on Imaging Sciences, 17(4), 2110-2141, 2024.

    Copyright (c) 2024, Zhiyuan Lyu, Lok Ming Lui, Gary P. T. Choi
    https://github.com/garyptchoi/spherical-density-equalizing-map
    """
    # Extract landmark coordinates
    S = v.copy()  # (nv, 3)
    S_landmark = S[landmark, :]  # (k, 3)

    # Rotation matrices as functions of angles
    def R_x(t):
        return np.array([
            [1, 0, 0],
            [0, np.cos(t), -np.sin(t)],
            [0, np.sin(t), np.cos(t)]
        ])

    def R_y(g):
        return np.array([
            [np.cos(g), 0, np.sin(g)],
            [0, 1, 0],
            [-np.sin(g), 0, np.cos(g)]
        ])

    def R_z(h):
        return np.array([
            [np.cos(h), -np.sin(h), 0],
            [np.sin(h), np.cos(h), 0],
            [0, 0, 1]
        ])

    # Derivatives of rotation matrices
    def dR_x(t):
        return np.array([
            [0, 0, 0],
            [0, -np.sin(t), -np.cos(t)],
            [0, np.cos(t), -np.sin(t)]
        ])

    def dR_y(g):
        return np.array([
            [-np.sin(g), 0, np.cos(g)],
            [0, 0, 0],
            [-np.cos(g), 0, -np.sin(g)]
        ])

    def dR_z(h):
        return np.array([
            [-np.sin(h), -np.cos(h), 0],
            [np.cos(h), -np.sin(h), 0],
            [0, 0, 0]
        ])

    # Landmark mismatch error function
    def L(w):
        t, g, h = w
        rotated = (R_x(t) @ R_y(g) @ R_z(h) @ S_landmark.T).T  # (k, 3)
        return np.sum((rotated - target) ** 2)

    # Initialization
    E = 1.0
    para_t = 0.0
    para_g = 0.0
    para_h = 0.0
    dt = 0.001
    step = 0
    L_old = L([para_t, para_g, para_h])

    # Gradient descent loop
    while E > 1e-6 and step < 1000:
        # Compute rotated landmarks and their derivatives
        R = R_x(para_t) @ R_y(para_g) @ R_z(para_h)  # Combined rotation matrix
        rotated = (R @ S_landmark.T).T  # (k, 3)

        # Gradient computation for each parameter
        grad_t = np.sum(2 * (rotated - target) *
                        (dR_x(para_t) @ R_y(para_g) @ R_z(para_h) @ S_landmark.T).T)
        grad_g = np.sum(2 * (rotated - target) *
                        (R_x(para_t) @ dR_y(para_g) @ R_z(para_h) @ S_landmark.T).T)
        grad_h = np.sum(2 * (rotated - target) *
                        (R_x(para_t) @ R_y(para_g) @ dR_z(para_h) @ S_landmark.T).T)

        # Update parameters
        para_t -= dt * grad_t
        para_g -= dt * grad_g
        para_h -= dt * grad_h

        # Update landmark mismatch error
        L_temp = L([para_t, para_g, para_h])
        E = abs(L_temp - L_old)
        L_old = L_temp

        step += 1

    # Apply final rotation to all vertices
    final_rotation = R_x(para_t) @ R_y(para_g) @ R_z(para_h)
    v_rotated = (final_rotation @ S.T).T  # (nv, 3)

    return v_rotated