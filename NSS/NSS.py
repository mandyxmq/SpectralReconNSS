import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from scipy import interpolate
from scipy.optimize import lsq_linear
from util import *
import pandas as pd

def sample_feasible_z_using_bounds(c0, N, v, num_samples=100000):
    d = N.shape[1]
    """
    Samples multiple spectra satisfying RGB constraint by computing valid bounds for z.

    Args:
        c0: base solution for B-spline coefficients, shape (n,)
        N: null space of RGB projection matrix times B-spline basis, shape (n, d)
        v: target RGB values, shape (3,)
        num_samples: how many random z samples to try

    Returns:
        z_samples: array of shape (m, d) of feasible z samples
        c_samples: array of shape (m, n) of corresponding B-spline coefficients
    """

    # Step 1: compute valid z bounds so that 0 <= c0 + N @ z <= 1
    bounds = []
    for i in range(d):
        valid = N[:, i] != 0
        z_lower = np.where(valid, -c0 / N[:, i], -np.inf)
        z_upper = np.where(valid, (1 - c0) / N[:, i], np.inf)
        zmin = np.max(np.where(valid, np.minimum(z_lower, z_upper), -np.inf))
        zmax = np.min(np.where(valid, np.maximum(z_lower, z_upper), np.inf))
        zmin = max(zmin, -1e3)
        zmax = min(zmax, 1e3)
        if zmax < zmin:
            raise ValueError("No feasible region for z.")
        bounds.append((zmin, zmax))

    # Step 2: sample uniformly within bounds
    z_samples = np.array([
        np.random.uniform(low, high, size=num_samples)
        for (low, high) in bounds
    ]).T

    # Step 3: check if c is in bounds
    c_samples = c0[None, :] + z_samples @ N.T
    mask = np.all((c_samples >= 0) & (c_samples <= 1), axis=1)
    feasible_z = z_samples[mask]
    feasible_c = c_samples[mask]

    return feasible_z, feasible_c


def sample_feasible_z_with_end_constraints(c0, N, knots, degree, t_start, v_start, t_end, v_end,
    num_samples=1000000, top_k=10000, lambda_bc=10.0):
    """
    Samples multiple spectra satisfying RGB constraint and selects those that best match
    boundary conditions at the start and end of the B-spline.

    Args:
        c0: base solution for B-spline coefficients, shape (n,)
        N: null space of RGB projection matrix times B-spline basis, shape (n, d)
        knots: knot vector for B-spline
        degree: degree of B-spline
        t_start: parameter value at start boundary
        v_start: target value at start boundary
        t_end: parameter value at end boundary
        v_end: target value at end boundary
        num_samples: how many random z samples to try
        top_k: number of best solutions to return
        lambda_bc: weight for boundary condition loss

    Returns:
        z_samples: array of shape (top_k, d) of best z samples
        c_samples: array of shape (top_k, n) of corresponding B-spline coefficients
    """

    d = N.shape[1]
    # Step 1: compute valid z bounds so that 0 <= c <= 1
    bounds = []
    for i in range(d):
        valid = N[:, i] != 0
        z_lower = np.where(valid, -c0 / N[:, i], -np.inf)
        z_upper = np.where(valid, (1 - c0) / N[:, i], np.inf)
        zmin = np.max(np.where(valid, np.minimum(z_lower, z_upper), -np.inf))
        zmax = np.min(np.where(valid, np.maximum(z_lower, z_upper), np.inf))
        zmin = max(zmin, -1e3)
        zmax = min(zmax, 1e3)
        if zmax < zmin:
            raise ValueError("No feasible region for z.")
        bounds.append((zmin, zmax))

    # Step 2: sample uniformly within bounds
    z_samples = np.array([
        np.random.uniform(low, high, size=num_samples)
        for (low, high) in bounds
    ]).T

    # Step 3: compute corresponding c
    c_samples = c0[None, :] + z_samples @ N.T

    # Step 4: filter for 0 <= c <= 1
    mask = np.all((c_samples >= 0) & (c_samples <= 1), axis=1)
    c_feasible = c_samples[mask]
    z_feasible = z_samples[mask]

    if len(z_feasible) == 0:
        return np.empty((0, d))

    # Step 5: evaluate B-spline endpoints
    spline_vals_start = np.array([
        BSpline(knots, c, degree)(t_start) for c in c_feasible
    ])
    spline_vals_end = np.array([
        BSpline(knots, c, degree)(t_end) for c in c_feasible
    ])

    # Step 6: compute soft constraint loss
    loss = (
        lambda_bc * ((spline_vals_start - v_start)**2 + (spline_vals_end - v_end)**2)
    )

    # Step 7: sort and keep top_k samples with smallest loss
    top_indices = np.argsort(loss)[:top_k]
    return z_feasible[top_indices], c_feasible[top_indices]
    

def sample_joint_solutions(M, B, y3, c0, s2, N, num_samples=100000, top_k=1000):
    """
    Samples multiple spectra satisfying RGB constraint and selects those that best match y3
    when multiplied with a fixed second spectrum s2.

    Args:
        M: RGB projection matrix, shape (3, m)
        B: B-spline basis matrix, shape (m, n)
        y3: target RGB of the product spectrum, shape (3,)
        c0: base solution for first spectrum, shape (n,)
        s2: second spectrum (fixed), shape (m,)
        N: null space of M @ B, shape (n, d)
        num_samples: how many random z samples to try
        top_k: number of best solutions to return

    Returns:
        List of (c, z, loss) tuples sorted by loss
    """
    d = N.shape[1]
    results = []

    # Step 1: Compute valid bounds for z so that 0 <= c = c0 + N @ z <= 1
    bounds = []
    for i in range(d):
        valid = N[:, i] != 0
        z_lower = np.where(valid, -c0 / N[:, i], -np.inf)
        z_upper = np.where(valid, (1 - c0) / N[:, i], np.inf)
        zmin = np.max(np.where(valid, np.minimum(z_lower, z_upper), -np.inf))
        zmax = np.min(np.where(valid, np.maximum(z_lower, z_upper), np.inf))
        zmin = max(zmin, -1e3)
        zmax = min(zmax, 1e3)
        if zmax < zmin:
            raise ValueError("Infeasible bounds for z")
        bounds.append((zmin, zmax))

    # Step 2: Sample from feasible region
    for _ in range(num_samples):
        z = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        c = c0 + N @ z
        if not np.all((c >= 0) & (c <= 1)):
            continue
        s1 = B @ c
        s3 = s1 * s2
        y3_pred = M @ s3
        loss = np.sum((y3_pred - y3) ** 2)
        results.append((c, z, loss))

    # Step 3: Return top_k results by loss
    results.sort(key=lambda tup: tup[2])
    return results[:top_k]


def build_bspline_basis(wavelengths, low, high, num_knots, degree=3):
    """
    Build normalized B-spline basis values over wavelengths.

    Returns
    -------
    basis_values : (n_basis, n_wavelengths) ndarray
    knots        : (n_knots + 2*degree,) ndarray
    """
    # interior knots
    knots_internal = np.linspace(low, high, num_knots)
    # add endpoint multiplicities for clamping
    knots = np.concatenate(([low] * degree, knots_internal, [high] * degree))

    n_basis = len(knots) - degree - 1
    basis_functions = [
        BSpline(knots, (np.arange(len(knots) - 1) == i).astype(int), degree)
        for i in range(n_basis)
    ]

    basis_values = np.array([b(wavelengths) for b in basis_functions])

    # Normalize to sum to 1 at each wavelength
    basis_values /= basis_values.sum(axis=0, keepdims=True)

    return basis_values, knots


def build_M_matrix(wavelengths, delta_lambda):
    """
    Build the RGB projection matrix M from wavelength to sRGB.
    """
    XYZ = cie1931_xyz(wavelengths)
    M_xyz = np.stack([XYZ[0], XYZ[1], XYZ[2]], axis=0) * delta_lambda

    M_xyz_to_srgb = np.array([
        [ 3.240479, -1.537150, -0.498535 ],
        [-0.969256,  1.875991,  0.041556 ],
        [ 0.055648, -0.204043,  1.057311 ]
    ])

    return M_xyz_to_srgb @ M_xyz


def fit_and_sample(M, basis_values, rgb, num_samples):
    """
    Fit B-spline coefficients to match target rgb and sample feasible spectra.
    """
    B = basis_values.T
    A = M @ B

    # particular solution
    c0 = lsq_linear(A, rgb, bounds=(0, 1)).x

    # null space of A
    N = null_space(A)
    BN = B @ N 

    # sensitivity weight per wavelength
    weight = np.sum(M * M, axis=0)
    v = np.sum(weight[:, None] * BN**2, axis=0)

    # sample feasible z in null space coords, get coefficients
    feasible_z, c_samples = sample_feasible_z_using_bounds(
        c0, N, v, num_samples=num_samples
    )

    # spectra samples: (num_samples, n_wavelengths)
    spectra_samples = basis_values.T @ c_samples.T
    spectra_samples = spectra_samples.T

    mean_spectrum = np.mean(spectra_samples, axis=0)
    std_spectrum = np.std(spectra_samples, axis=0)

    return c0, N, v, c_samples, spectra_samples, mean_spectrum, std_spectrum


def adaptive_spectrum_sampling(paint_wavelengths, spectra, low=400.0, high=700.0, wavelengthnum=400, 
                               knotnum_coarse=3, knotnum_fine=8, num_samples=100000):
    """
    Perform adaptive spectrum sampling to generate spectra matching target RGB.

    Args:
        paint_wavelengths: wavelengths of input spectrum
        spectra: input spectrum values
        low: lower wavelength bound
        high: upper wavelength bound
        wavelengthnum: number of wavelength samples
        knotnum_coarse: number of knots for coarse B-spline basis
        knotnum_fine: number of knots for fine B-spline basis
        num_samples: number of spectrum samples to generate
    Returns:
        spectra_samples_coarse: sampled spectra from coarse basis
        spectra_samples_fine: sampled spectra from fine basis
        spectra_samples_adaptive: sampled spectra from adaptive method
        mean_spectrum_coarse: mean spectrum from coarse samples
        std_spectrum_coarse: std dev spectrum from coarse samples
        mean_spectrum_fine: mean spectrum from fine samples
        std_spectrum_fine: std dev spectrum from fine samples
        mean_spectrum_adaptive: mean spectrum from adaptive samples
        std_spectrum_adaptive: std dev spectrum from adaptive samples
    """
    # Step 1: Resample the input spectrum onto a regular wavelength grid
    wavelengths = np.linspace(low, high, wavelengthnum)
    delta_lambda = wavelengths[1] - wavelengths[0]

    f_interp = interpolate.interp1d(paint_wavelengths, spectra, fill_value="extrapolate")
    spectra_resampled = f_interp(wavelengths)

    # Step 2: Convert target spectrum to RGB & build M
    xyz = spectrum_to_xyz(spectra_resampled, wavelengths)
    rgb = xyz_to_srgb(xyz[0], xyz[1], xyz[2])

    M = build_M_matrix(wavelengths, delta_lambda)

    # Step 3: Coarse sampling
    basis_values_coarse, knots_coarse = build_bspline_basis(
        wavelengths, low, high, knotnum_coarse, degree=3
    )

    c0_coarse, N_coarse, v_coarse, c_samples_coarse, spectra_samples_coarse, \
        mean_spectrum_coarse, std_spectrum_coarse = fit_and_sample(
            M, basis_values_coarse, rgb, num_samples
        )

    # Step 4: Fine sampling
    basis_values_fine, knots_fine = build_bspline_basis(
        wavelengths, low, high, knotnum_fine, degree=3
    )

    c0_fine, N_fine, v_fine, c_samples_fine, spectra_samples_fine, \
        mean_spectrum_fine, std_spectrum_fine = fit_and_sample(
            M, basis_values_fine, rgb, num_samples
        )

    # Step 5: Adaptive sampling: sample only in residual subspace
    B_coarse = basis_values_coarse.T  # (n_wavelengths, n_basis_coarse)
    B_fine   = basis_values_fine.T    # (n_wavelengths, n_basis_fine)

    # Map coarse coefficients to fine basis via least squares:
    T = np.linalg.lstsq(B_fine, B_coarse, rcond=None)[0]

    # Projection onto the coarse subspace inside the fine coefficient space
    P = T @ np.linalg.pinv(T)                 
    R = np.eye(P.shape[0]) - P                

    A_new = M @ B_fine
    N_f = null_space(A_new) 
    N_residual = R @ N_f

    # Lift coarse particular solution into fine basis
    c0_lifted = T @ c0_coarse                 # (n_basis_fine,)

    # Use v from fine step (same dimensionality as N_f)
    feasible_z_adaptive, c_samples_adaptive = sample_feasible_z_using_bounds(
        c0_lifted, N_residual, v_fine, num_samples=num_samples
    )

    # Spectra from adaptive sampling (num_samples, n_wavelengths)
    spectra_samples_adaptive = basis_values_fine.T @ c_samples_adaptive.T
    spectra_samples_adaptive = spectra_samples_adaptive.T

    mean_spectrum_adaptive = np.mean(spectra_samples_adaptive, axis=0)
    std_spectrum_adaptive = np.std(spectra_samples_adaptive, axis=0)
    
    return spectra_samples_coarse, spectra_samples_fine, spectra_samples_adaptive, \
        mean_spectrum_coarse, std_spectrum_coarse, \
        mean_spectrum_fine, std_spectrum_fine, \
        mean_spectrum_adaptive, std_spectrum_adaptive