"""Poincaré section I/O for .npz files.

Standard file format (pyna convention):
  'R'        : ndarray, shape (n_lines, n_crossings) — major radius [m]
  'Z'        : ndarray, shape (n_lines, n_crossings) — vertical coord [m]
  'phi'      : float or ndarray — toroidal angle(s) of the section [rad]
  'metadata' : dict (optional) — equilibrium name, date, config, etc.

This format is also compatible with Juna.jl notebook output
(display_Poincare_plot_in_npz.ipynb), where each key is the phi index
and each value has shape (n_crossings, 3) with columns [R, Z, phi].
The load function auto-detects both formats.

No copyrighted W7-X data is included in this module; only generic I/O logic.
"""

import numpy as np
from typing import Optional, Union


def save_poincare_npz(path: str,
                      R_arr: np.ndarray,
                      Z_arr: np.ndarray,
                      phi: Union[float, np.ndarray],
                      metadata: Optional[dict] = None) -> None:
    """Save Poincaré section data to a compressed .npz file.

    Parameters
    ----------
    path : str
        Output file path (will add .npz extension if missing).
    R_arr : ndarray, shape (n_lines, n_crossings)
        R coordinates of Poincaré crossings.
    Z_arr : ndarray, shape (n_lines, n_crossings)
        Z coordinates of Poincaré crossings.
    phi : float or ndarray
        Toroidal angle(s) of the Poincaré section [rad].
    metadata : dict, optional
        Arbitrary key-value metadata (stored as a pickled object array).
    """
    save_dict = dict(R=R_arr, Z=Z_arr, phi=np.asarray(phi))
    if metadata is not None:
        # store as 0-d object array so np.load can retrieve it
        meta_arr = np.empty(1, dtype=object)
        meta_arr[0] = metadata
        save_dict['metadata'] = meta_arr
    np.savez_compressed(path, **save_dict)


def load_poincare_npz(path: str) -> dict:
    """Load Poincaré section data from a .npz file.

    Supports two formats:

    1. **pyna format**: keys 'R', 'Z', 'phi', optional 'metadata'.
    2. **Juna.jl format**: integer-keyed sections, each shape (n, 3)
       with columns [R, Z, phi_val].

    Returns
    -------
    dict with keys:
      'R'        : ndarray (n_lines, n_crossings)
      'Z'        : ndarray (n_lines, n_crossings)
      'phi'      : float or ndarray
      'metadata' : dict or None
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())

    # Detect pyna format
    if 'R' in keys and 'Z' in keys:
        metadata = None
        if 'metadata' in keys:
            m = d['metadata']
            if m.dtype == object and m.size == 1:
                metadata = m[0]
            else:
                metadata = m
        return {
            'R': d['R'],
            'Z': d['Z'],
            'phi': d['phi'] if 'phi' in keys else np.nan,
            'metadata': metadata,
        }

    # Detect Juna.jl format: keys are integers (as strings), values shape (n, 3)
    try:
        int_keys = sorted(int(k) for k in keys)
    except ValueError:
        raise ValueError(f"Unrecognised .npz format; found keys: {keys}")

    # Stack all lines into (n_lines, n_crossings) arrays
    R_list, Z_list = [], []
    phi_vals = []
    for k in int_keys:
        arr = d[str(k)]  # shape (n_crossings, 3): [R, Z, phi]
        R_list.append(arr[:, 0])
        Z_list.append(arr[:, 1])
        phi_vals.append(arr[0, 2] if arr.shape[1] > 2 else np.nan)

    R_arr = np.array(R_list)
    Z_arr = np.array(Z_list)

    return {
        'R': R_arr,
        'Z': Z_arr,
        'phi': np.array(phi_vals),
        'metadata': None,
    }


def plot_poincare_from_npz(path: str, ax=None, **scatter_kwargs):
    """Load and plot a Poincaré section from a .npz file.

    Parameters
    ----------
    path : str
        Path to .npz file.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if None.
    **scatter_kwargs
        Extra keyword arguments passed to ax.scatter().

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    data = load_poincare_npz(path)
    R = data['R']   # (n_lines, n_crossings)
    Z = data['Z']

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))

    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')

    scatter_defaults = dict(s=1.0, marker='.', alpha=0.6)
    scatter_defaults.update(scatter_kwargs)

    for i in range(R.shape[0]):
        ax.scatter(R[i], Z[i], **scatter_defaults)

    return ax
