"""Rotation utilities for spherical functions.

This module provides functions for rotating spherical functions in spatial domain
and generating random SO(3) rotation matrices.
"""

import torch
import torch.nn.functional as F


def random_rotation_matrix(device: torch.device | None = None) -> torch.Tensor:
    """Generate a random SO(3) rotation matrix using QR decomposition.

    Uses the QR decomposition method to generate a uniformly distributed
    random rotation matrix from the Haar measure on SO(3).

    Args:
        device: Device to create the tensor on. Defaults to CPU.

    Returns:
        torch.Tensor: A 3x3 rotation matrix (orthogonal with determinant +1).
    """
    # Generate random 3x3 matrix with standard normal entries
    random_matrix = torch.randn(3, 3, device=device, dtype=torch.float64)

    # QR decomposition
    q, r = torch.linalg.qr(random_matrix)

    # Ensure determinant is +1 (proper rotation, not reflection)
    # The QR decomposition can give det(Q) = -1, so we fix this
    d = torch.diag(r)
    sign_d = torch.sign(d)
    # Handle zero diagonal elements (very rare)
    sign_d = torch.where(sign_d == 0, torch.ones_like(sign_d), sign_d)
    q = q * sign_d.unsqueeze(0)

    # Ensure determinant is +1
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]

    return q


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        theta: Colatitude angle in [0, pi]. Shape (...,).
        phi: Longitude angle in [0, 2*pi). Shape (...,).

    Returns:
        torch.Tensor: Cartesian coordinates of shape (..., 3) with (x, y, z).
    """
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
        xyz: Cartesian coordinates of shape (..., 3).

    Returns:
        Tuple of (theta, phi) where:
            theta: Colatitude in [0, pi].
            phi: Longitude in [0, 2*pi).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # Compute r (should be 1 for unit sphere, but normalize for safety)
    r = torch.sqrt(x**2 + y**2 + z**2)
    r = torch.clamp(r, min=1e-10)  # Avoid division by zero

    # Colatitude: theta = arccos(z/r)
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))

    # Longitude: phi = atan2(y, x), shifted to [0, 2*pi)
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)

    return theta, phi


def create_spherical_grid(
    nlat: int, nlon: int, grid: str = 'equiangular', device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create spherical grid coordinates.

    Args:
        nlat: Number of latitude points.
        nlon: Number of longitude points.
        grid: Grid type. Currently only 'equiangular' is supported.
        device: Device to create tensors on.

    Returns:
        Tuple of (theta, phi) meshgrids, each of shape (nlat, nlon).
    """
    if grid == 'equiangular':
        # Equiangular grid: theta from 0 to pi, phi from 0 to 2*pi (exclusive)
        theta = torch.linspace(0, torch.pi, nlat, device=device, dtype=torch.float64)
        phi = torch.linspace(0, 2 * torch.pi, nlon + 1, device=device, dtype=torch.float64)[:-1]
    else:
        raise ValueError(f"Unsupported grid type: {grid}")

    # Create meshgrid
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    return theta_grid, phi_grid


def rotate_spherical_function(
    f_grid: torch.Tensor,
    rotation_matrix: torch.Tensor,
    grid: str = 'equiangular',
) -> torch.Tensor:
    """Rotate a spherical function by sampling at rotated grid points.

    Applies an SO(3) rotation R to a function f on the sphere by computing:
        (R * f)(x) = f(R^{-1} x)

    This is done by:
    1. For each output grid point x, compute R^{-1} x
    2. Find the corresponding spherical coordinates (theta', phi')
    3. Interpolate f at (theta', phi')

    Args:
        f_grid: Function values on the sphere of shape (batch, nlat, nlon).
        rotation_matrix: 3x3 SO(3) rotation matrix.
        grid: Grid type used ('equiangular').

    Returns:
        Rotated function on the same grid, shape (batch, nlat, nlon).
    """
    batch_size, nlat, nlon = f_grid.shape
    device = f_grid.device
    dtype = f_grid.dtype

    # Create output grid coordinates
    theta_grid, phi_grid = create_spherical_grid(nlat, nlon, grid=grid, device=device)

    # Convert grid to Cartesian coordinates
    xyz = spherical_to_cartesian(theta_grid, phi_grid)  # (nlat, nlon, 3)

    # Apply inverse rotation: R^{-1} = R^T for rotation matrices
    rotation_matrix = rotation_matrix.to(device=device, dtype=torch.float64)
    r_inv = rotation_matrix.T

    # Rotate all points: (nlat, nlon, 3) @ (3, 3) -> (nlat, nlon, 3)
    xyz_rotated = torch.einsum('ijk,lk->ijl', xyz, r_inv)

    # Convert back to spherical coordinates
    theta_new, phi_new = cartesian_to_spherical(xyz_rotated)

    # For grid_sample, we need normalized coordinates in [-1, 1]
    # The input image has theta on the height axis (0 to pi) and phi on width axis (0 to 2*pi)
    # grid_sample expects (x, y) where x is width and y is height
    
    # Pad the function for periodic boundary conditions in phi
    # Wrap around by appending first columns to the end and last columns to the beginning
    pad_width = 4
    f_input = f_grid.to(torch.float64).unsqueeze(1)  # (batch, 1, nlat, nlon)
    f_padded = F.pad(f_input, (pad_width, pad_width, 0, 0), mode='circular')
    
    # Adjust phi to account for padding
    # Original phi range: [0, 2*pi) mapped to pixel indices [0, nlon)
    # With padding: pixel indices become [pad_width, nlon + pad_width)
    # We need to map phi to the padded coordinate system
    
    nlon_padded = nlon + 2 * pad_width
    
    # Convert theta and phi to normalized grid coordinates for the padded image
    # theta: [0, pi] -> pixel [0, nlat-1] -> normalized [-1, 1]
    # phi: [0, 2*pi) -> pixel [0, nlon-1] -> with padding -> normalized
    
    # Pixel coordinates (before normalization)
    theta_pixel = theta_new / torch.pi * (nlat - 1)
    phi_pixel = phi_new / (2 * torch.pi) * nlon + pad_width
    
    # Normalize to [-1, 1] for grid_sample with align_corners=True
    # For align_corners=True: pixel 0 -> -1, pixel (size-1) -> 1
    theta_norm = 2.0 * theta_pixel / (nlat - 1) - 1.0
    phi_norm = 2.0 * phi_pixel / (nlon_padded - 1) - 1.0

    # Create sampling grid: (batch, nlat, nlon, 2) with (x=phi, y=theta)
    sample_grid = torch.stack([phi_norm, theta_norm], dim=-1)
    sample_grid = sample_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Use grid_sample for bilinear interpolation
    f_rotated = F.grid_sample(
        f_padded.float(),
        sample_grid.float(),
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    # Remove channel dimension and convert back to original dtype
    f_rotated = f_rotated.squeeze(1).to(dtype)

    return f_rotated
