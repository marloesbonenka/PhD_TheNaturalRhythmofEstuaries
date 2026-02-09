"""Functions for computing morphological activity metrics.

Includes:
- Cumulative activity (Σ|Δz|)
- Morph-time conversions
- Activity plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def cumulative_activity(profiles_time_space: np.ndarray) -> np.ndarray:
    """Compute cumulative activity Σ|Δz| along the time axis.
    
    Parameters
    ----------
    profiles_time_space : np.ndarray
        2D array of shape (time, space) with bed levels.
        
    Returns
    -------
    np.ndarray
        Cumulative activity array, same shape (time, space).
    """
    z = np.asarray(profiles_time_space)
    if z.ndim != 2:
        raise ValueError("Expected 2D array (time, space)")
    dz = np.diff(z, axis=0)
    abs_dz = np.abs(dz)
    zeros = np.zeros((1, z.shape[1]))
    abs_dz0 = np.vstack([zeros, abs_dz])
    return np.cumsum(abs_dz0, axis=0)


def morph_years_from_datetimes(times: pd.DatetimeIndex, *, startdate=None, morfac=1.0) -> np.ndarray:
    """Convert datetime series to morphological years.
    
    Parameters
    ----------
    times : pd.DatetimeIndex
        Series of timestamps.
    startdate : str or pd.Timestamp, optional
        Reference start date. If None, uses first timestamp.
    morfac : float
        Morphological acceleration factor.
        
    Returns
    -------
    np.ndarray
        Morphological years from start.
    """
    if startdate is None:
        t0 = times[0]
    else:
        t0 = pd.Timestamp(startdate)
    hydro_years = np.array([(t - t0).total_seconds() / (365.25 * 24 * 3600) for t in times])
    return hydro_years * float(morfac)


def plot_activity_and_first_profile(
    *,
    dist_m: np.ndarray,
    first_profile: np.ndarray,
    cumact: np.ndarray,
    morph_years: np.ndarray,
    title: str,
    outpath: Path,
    show: bool = False,
):
    """Plot cumulative activity heatmap with initial bed profile.
    
    Parameters
    ----------
    dist_m : np.ndarray
        Distance along cross-section in meters.
    first_profile : np.ndarray
        Initial bed level profile.
    cumact : np.ndarray
        Cumulative activity array (time, space).
    morph_years : np.ndarray
        Morphological years for y-axis.
    title : str
        Plot title.
    outpath : Path
        Output file path.
    show : bool
        If True, display plot interactively.
    """
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    x_km = dist_m / 1000.0
    extent = [x_km.min(), x_km.max(), morph_years.min(), morph_years.max()]
    vmax = np.nanpercentile(cumact, 98) if np.any(np.isfinite(cumact)) else 1.0

    im = ax0.imshow(
        cumact,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=0,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Sigma |\Delta z_b|$ [m]", fontsize=11, fontweight='bold')

    ax0.set_ylabel('Morphological time [years]', fontsize=11, fontweight='bold')
    ax0.set_title(title, fontsize=12, fontweight='bold')

    ax1.plot(x_km, first_profile, color='black', linewidth=2)
    ax1.set_xlabel('Cross-section distance [km]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bed level [m]', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
