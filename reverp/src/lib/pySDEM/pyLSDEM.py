import numpy as np
from scipy.sparse.linalg import spsolve
from reverp.src.lib.pySDEM.regular_triangle import regular_triangle
from reverp.src.lib.pySDEM.face_area import face_area
from reverp.src.lib.pySDEM.f2v_area import f2v_area
from reverp.src.lib.pySDEM.update_and_correct_overlap import update_and_correct_overlap
from reverp.src.lib.pySDEM.compute_gradient_3D import compute_gradient_3D
from reverp.src.lib.pySDEM.laplace_beltrami import laplace_beltrami
from reverp.src.lib.pySDEM.lumped_mass_matrix import lumped_mass_matrix
from reverp.src.lib.pySDEM.spherical_conformal_map import spherical_conformal_map
from reverp.src.lib.pySDEM.mobius_area_correction_spherical import mobius_area_correction_spherical
from reverp.src.lib.pySDEM.optimal_rotation import optimal_rotation

def pyLSDEM(v, f, population, S=None, landmark=None, target=None,
            alpha=1.0, beta=1.0, gamma=1.0, dt=0.01, epsilon=1e-3, max_iter=200):
    # Input validation
    if alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("Weighting parameters must be nonnegative.")
    if target is not None and np.max(np.abs(np.sqrt(np.sum(target**2, axis=1)) - 1)) > 1e-4:
        raise ValueError("Target positions must lie on the unit sphere.")

    # Initial spherical parameterization
    if S is None:
        S1 = spherical_conformal_map(v, f)
        S, _ = mobius_area_correction_spherical(v, f, S1)

    # Normalize S
    r = np.array([
        S[:, 0] / np.sqrt(np.sum(S**2, axis=1)),
        S[:, 1] / np.sqrt(np.sum(S**2, axis=1)),
        S[:, 2] / np.sqrt(np.sum(S**2, axis=1))
    ]).T

    bigtri = regular_triangle(f, r)

    # Normalize population and compute initial density
    population = population / np.sum(population)
    rho_f = population / face_area(f, r)
    rho_v = f2v_area(r, f) * rho_f

    step = 0
    f_diff = np.inf
    print("Step     ||f_n - f_{n-1}||")

    r_old = r.copy()
    while f_diff >= epsilon and step < max_iter:
        # Optimal rotation for landmark alignment
        if landmark is not None and target is not None:
            r = optimal_rotation(r, landmark, target)

        # Update density
        L = laplace_beltrami(r, f)
        A = lumped_mass_matrix(r, f)
        rho_v_temp = spsolve(A + dt * L, A @ rho_v)

        # Density term (dE1)
        grad_rho_temp_f = compute_gradient_3D(r, f, rho_v_temp)
        grad_rho_temp_v = f2v_area(r, f) * grad_rho_temp_f
        dr = -np.column_stack((
            grad_rho_temp_v[:, 0] / rho_v_temp,
            grad_rho_temp_v[:, 1] / rho_v_temp,
            grad_rho_temp_v[:, 2] / rho_v_temp
        ))
        dE1 = alpha * (dr - np.sum(dr * r, axis=1)[:, np.newaxis] * r)

        # Harmonic term (dE2)
        Lr = -spsolve(A, L @ r)
        dE2 = -beta * (Lr - np.sum(Lr * r, axis=1)[:, np.newaxis] * r)

        # Landmark mismatch term (dE3)
        dE3 = np.zeros_like(r)
        if landmark is not None and target is not None:
            dE3[landmark, :] = -gamma * (r[landmark, :] - target)

        # Combine terms
        dE = dE1 + dE2 + dE3

        # Update and correct overlap
        r = update_and_correct_overlap(f, S, r, bigtri, dE, dt)

        # Compute step difference
        step += 1
        f_diff = np.max(np.abs(r - r_old))
        print(f"{step}        {f_diff}")

        # Update density
        rho_f = population / face_area(f, r)
        rho_v = f2v_area(r, f) * rho_f
        r_old = r.copy()

    return r