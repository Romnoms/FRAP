#!/usr/bin/env python3
"""
FRAP Analysis Script - Publication Standard
============================================

This script performs Fluorescence Recovery After Photobleaching (FRAP) analysis
following publication standards for high-impact journals.

Implements:
- Double normalization (Phair et al., 2004)
- Full-scale normalization (Ellenberg et al., 1997)
- Single and double exponential curve fitting
- Automatic ROI detection
- Quality control metrics

Usage:
    python frap_analysis.py --input data.lif --output results/

Or import as module:
    from frap_analysis import FRAPAnalyzer
    analyzer = FRAPAnalyzer()
    results = analyzer.analyze_lif('data.lif', series_pairs=[(9, 10)])

Requirements:
    pip install readlif numpy scipy matplotlib pandas

References:
    - Phair RD et al. (2004) Methods Enzymol 375:393-414
    - Ellenberg J et al. (1997) J Cell Biol 138:1271-1287
    - Roca-Cusachs P et al. (2013) Methods Cell Biol 116:271-291

Author: Generated with Claude Code
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
import argparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json

# Optional imports
try:
    from readlif.reader import LifFile
    HAS_READLIF = True
except ImportError:
    HAS_READLIF = False
    print("Warning: readlif not installed. LIF file support disabled.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ROIData:
    """Container for ROI masks and statistics"""
    bleach_mask: np.ndarray
    reference_mask: np.ndarray
    background_mask: np.ndarray
    bleach_pixels: int = 0
    reference_pixels: int = 0
    background_pixels: int = 0

    def __post_init__(self):
        self.bleach_pixels = np.sum(self.bleach_mask)
        self.reference_pixels = np.sum(self.reference_mask)
        self.background_pixels = np.sum(self.background_mask)


@dataclass
class FRAPFitResult:
    """Container for curve fitting results"""
    model_type: str  # 'single' or 'double'
    mobile_fraction: float
    immobile_fraction: float
    tau: float  # primary time constant
    t_half: float  # half-time of recovery
    r_squared: float
    aic: float
    parameters: dict = field(default_factory=dict)
    fitted_curve: np.ndarray = None

    def to_dict(self) -> dict:
        return {
            'model_type': self.model_type,
            'mobile_fraction': self.mobile_fraction,
            'immobile_fraction': self.immobile_fraction,
            'tau_s': self.tau,
            't_half_s': self.t_half,
            'r_squared': self.r_squared,
            'aic': self.aic,
            **self.parameters
        }


@dataclass
class FRAPResult:
    """Complete FRAP analysis result"""
    name: str
    times: np.ndarray
    raw_bleach_intensities: np.ndarray
    raw_reference_intensities: np.ndarray
    background_intensity: float
    double_normalized: np.ndarray
    full_scale_normalized: np.ndarray
    fit_single: Optional[FRAPFitResult] = None
    fit_double: Optional[FRAPFitResult] = None
    best_fit: Optional[FRAPFitResult] = None
    roi_data: Optional[ROIData] = None
    time_interval: float = 1.0
    n_frames: int = 0
    quality_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_frames = len(self.times)
        # Select best fit based on AIC
        if self.fit_single and self.fit_double:
            # Prefer simpler model unless double is significantly better (ΔAIC > 2)
            if self.fit_double.aic < self.fit_single.aic - 2:
                self.best_fit = self.fit_double
            else:
                self.best_fit = self.fit_single
        elif self.fit_single:
            self.best_fit = self.fit_single
        elif self.fit_double:
            self.best_fit = self.fit_double

    def to_dataframe(self) -> pd.DataFrame:
        """Export time-course data as DataFrame"""
        return pd.DataFrame({
            'time_s': self.times,
            'raw_bleach': self.raw_bleach_intensities,
            'raw_reference': self.raw_reference_intensities,
            'double_normalized': self.double_normalized,
            'full_scale_normalized': self.full_scale_normalized,
            'fitted': self.best_fit.fitted_curve if self.best_fit else np.nan
        })

    def summary_dict(self) -> dict:
        """Get summary statistics as dictionary"""
        result = {
            'name': self.name,
            'n_frames': self.n_frames,
            'time_interval_s': self.time_interval,
            'total_time_s': self.times[-1] if len(self.times) > 0 else 0,
            'background_intensity': self.background_intensity,
        }
        if self.best_fit:
            result.update(self.best_fit.to_dict())
        if self.roi_data:
            result['bleach_roi_pixels'] = self.roi_data.bleach_pixels
            result['reference_roi_pixels'] = self.roi_data.reference_pixels
        return result


# =============================================================================
# FRAP Models
# =============================================================================

def single_exponential(t: np.ndarray, mobile_fraction: float, tau: float) -> np.ndarray:
    """
    Single exponential FRAP recovery model.

    F(t) = Mf * (1 - exp(-t/τ))

    Parameters:
        t: Time points
        mobile_fraction: Fraction of molecules that recover (0-1)
        tau: Time constant of recovery

    Returns:
        Predicted normalized fluorescence
    """
    return mobile_fraction * (1 - np.exp(-t / tau))


def double_exponential(t: np.ndarray, mf_fast: float, tau_fast: float,
                       mf_slow: float, tau_slow: float) -> np.ndarray:
    """
    Double exponential FRAP recovery model (two populations).

    F(t) = Mf_fast * (1 - exp(-t/τ_fast)) + Mf_slow * (1 - exp(-t/τ_slow))

    Parameters:
        t: Time points
        mf_fast: Mobile fraction of fast component
        tau_fast: Time constant of fast component
        mf_slow: Mobile fraction of slow component
        tau_slow: Time constant of slow component

    Returns:
        Predicted normalized fluorescence
    """
    return mf_fast * (1 - np.exp(-t / tau_fast)) + mf_slow * (1 - np.exp(-t / tau_slow))


def diffusion_model(t: np.ndarray, D: float, r: float) -> np.ndarray:
    """
    Soumpasis diffusion model for circular bleach spot.

    F(t) = exp(-τD/2t) * (I0(τD/2t) + I1(τD/2t))

    where τD = r²/D is the characteristic diffusion time

    Parameters:
        t: Time points
        D: Diffusion coefficient (μm²/s)
        r: Radius of bleach spot (μm)

    Returns:
        Predicted normalized fluorescence
    """
    from scipy.special import i0, i1
    tau_D = r**2 / D
    x = tau_D / (2 * t + 1e-10)  # avoid division by zero
    return np.exp(-x) * (i0(x) + i1(x))


# =============================================================================
# ROI Detection
# =============================================================================

def detect_rois_auto(pre_frame: np.ndarray, post_frame: np.ndarray,
                     bleach_percentile: float = 85,
                     background_percentile: float = 5,
                     cell_percentile: float = 20,
                     buffer_pixels: int = 5) -> ROIData:
    """
    Automatically detect ROIs from pre/post bleach images.

    Parameters:
        pre_frame: Pre-bleach image
        post_frame: First post-bleach image
        bleach_percentile: Percentile threshold for bleach detection
        background_percentile: Percentile for background region
        cell_percentile: Percentile threshold for cell detection
        buffer_pixels: Dilation buffer around bleach ROI

    Returns:
        ROIData object with masks
    """
    diff = pre_frame.astype(float) - post_frame.astype(float)

    # Background ROI: lowest intensity pixels
    background_threshold = np.percentile(pre_frame, background_percentile)
    background_mask = pre_frame <= background_threshold

    # Cell mask: pixels above background
    cell_threshold = np.percentile(pre_frame, cell_percentile)
    cell_mask = pre_frame > cell_threshold

    # Bleach ROI: areas within cell that got significantly darker
    bleach_threshold = np.percentile(diff[cell_mask], bleach_percentile)
    bleach_mask = (diff > bleach_threshold) & cell_mask

    # Clean up bleach mask with morphological operations
    bleach_mask = ndimage.binary_opening(bleach_mask, iterations=2)
    bleach_mask = ndimage.binary_closing(bleach_mask, iterations=2)

    # Keep only largest connected component
    labeled, num_features = ndimage.label(bleach_mask)
    if num_features > 0:
        sizes = ndimage.sum(bleach_mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        bleach_mask = labeled == largest

    # Reference ROI: cell region minus dilated bleach area
    dilated_bleach = ndimage.binary_dilation(bleach_mask, iterations=buffer_pixels)
    reference_mask = cell_mask & ~dilated_bleach

    return ROIData(
        bleach_mask=bleach_mask,
        reference_mask=reference_mask,
        background_mask=background_mask
    )


def detect_rois_manual(image_shape: Tuple[int, int],
                       bleach_center: Tuple[int, int],
                       bleach_radius: int,
                       reference_region: Optional[Tuple[int, int, int, int]] = None,
                       background_region: Optional[Tuple[int, int, int, int]] = None) -> ROIData:
    """
    Create ROIs from manual specifications.

    Parameters:
        image_shape: (height, width) of images
        bleach_center: (y, x) center of bleach spot
        bleach_radius: Radius of circular bleach ROI
        reference_region: (y1, y2, x1, x2) bounding box for reference ROI
        background_region: (y1, y2, x1, x2) bounding box for background ROI

    Returns:
        ROIData object with masks
    """
    h, w = image_shape

    # Create circular bleach mask
    y, x = np.ogrid[:h, :w]
    cy, cx = bleach_center
    bleach_mask = (x - cx)**2 + (y - cy)**2 <= bleach_radius**2

    # Reference mask
    reference_mask = np.zeros((h, w), dtype=bool)
    if reference_region:
        y1, y2, x1, x2 = reference_region
        reference_mask[y1:y2, x1:x2] = True
        reference_mask[bleach_mask] = False  # Exclude bleach area

    # Background mask
    background_mask = np.zeros((h, w), dtype=bool)
    if background_region:
        y1, y2, x1, x2 = background_region
        background_mask[y1:y2, x1:x2] = True

    return ROIData(
        bleach_mask=bleach_mask,
        reference_mask=reference_mask,
        background_mask=background_mask
    )


# =============================================================================
# Normalization Functions
# =============================================================================

def normalize_double(frames: List[np.ndarray],
                     roi_data: ROIData,
                     pre_bleach_intensity: float,
                     pre_reference_intensity: float) -> np.ndarray:
    """
    Double normalization (Phair method).

    Corrects for:
    - Background fluorescence
    - Acquisition photobleaching (photofading)
    - Differences in starting intensity

    Formula:
    F_norm(t) = [(F_ROI(t) - F_bkgd) / (F_ref(t) - F_bkgd)] ×
                [(F_ref(i) - F_bkgd) / (F_ROI(i) - F_bkgd)]

    Parameters:
        frames: List of post-bleach frames
        roi_data: ROI masks
        pre_bleach_intensity: Pre-bleach intensity in bleach ROI
        pre_reference_intensity: Pre-bleach intensity in reference ROI

    Returns:
        Array of double-normalized intensity values
    """
    normalized = []

    # Background intensity (use first frame)
    F_bkgd = np.mean(frames[0].astype(float)[roi_data.background_mask]) \
             if np.any(roi_data.background_mask) else 0

    # Initial reference intensity
    F_ref_i = np.mean(frames[0].astype(float)[roi_data.reference_mask])

    for frame in frames:
        frame_f = frame.astype(float)

        F_ROI_t = np.mean(frame_f[roi_data.bleach_mask])
        F_ref_t = np.mean(frame_f[roi_data.reference_mask])

        # Double normalization formula
        numerator = (F_ROI_t - F_bkgd) * (F_ref_i - F_bkgd)
        denominator = (F_ref_t - F_bkgd) * (pre_bleach_intensity - F_bkgd)

        if denominator > 0:
            F_norm = numerator / denominator
        else:
            F_norm = 0

        normalized.append(F_norm)

    return np.array(normalized)


def normalize_full_scale(double_norm_values: np.ndarray,
                         pre_norm_value: float = 1.0) -> np.ndarray:
    """
    Full-scale normalization (Ellenberg method).

    Scales data so:
    - Pre-bleach = 1
    - Immediately post-bleach = 0

    Formula:
    F_fullscale(t) = (F_double(t) - F_double(0)) / (F_pre - F_double(0))

    Parameters:
        double_norm_values: Double-normalized intensity values
        pre_norm_value: Normalized pre-bleach value (typically 1.0)

    Returns:
        Array of full-scale normalized values
    """
    F_0 = double_norm_values[0]  # immediately post-bleach

    if (pre_norm_value - F_0) > 0:
        return (double_norm_values - F_0) / (pre_norm_value - F_0)
    else:
        return double_norm_values - F_0


# =============================================================================
# Curve Fitting
# =============================================================================

def fit_single_exponential(times: np.ndarray,
                           intensities: np.ndarray) -> Optional[FRAPFitResult]:
    """
    Fit single exponential recovery model.

    Parameters:
        times: Time points (seconds)
        intensities: Normalized fluorescence values

    Returns:
        FRAPFitResult or None if fitting fails
    """
    try:
        popt, pcov = curve_fit(
            single_exponential, times, intensities,
            p0=[0.7, 1.0],
            bounds=([0, 0.001], [1.5, 1000]),
            maxfev=10000
        )

        mobile_fraction, tau = popt
        fitted = single_exponential(times, *popt)

        # Calculate fit statistics
        ss_res = np.sum((intensities - fitted)**2)
        ss_tot = np.sum((intensities - np.mean(intensities))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(intensities)
        k = 2  # number of parameters
        aic = n * np.log(ss_res/n + 1e-10) + 2 * k

        return FRAPFitResult(
            model_type='single_exponential',
            mobile_fraction=mobile_fraction,
            immobile_fraction=1 - mobile_fraction,
            tau=tau,
            t_half=tau * np.log(2),
            r_squared=r_squared,
            aic=aic,
            fitted_curve=fitted
        )

    except Exception as e:
        warnings.warn(f"Single exponential fit failed: {e}")
        return None


def fit_double_exponential(times: np.ndarray,
                           intensities: np.ndarray) -> Optional[FRAPFitResult]:
    """
    Fit double exponential recovery model.

    Parameters:
        times: Time points (seconds)
        intensities: Normalized fluorescence values

    Returns:
        FRAPFitResult or None if fitting fails
    """
    if len(times) < 6:
        warnings.warn("Not enough data points for double exponential fit (need ≥6)")
        return None

    try:
        popt, pcov = curve_fit(
            double_exponential, times, intensities,
            p0=[0.3, 0.5, 0.3, 3.0],
            bounds=([0, 0.001, 0, 0.01], [1.0, 100, 1.0, 1000]),
            maxfev=10000
        )

        mf_fast, tau_fast, mf_slow, tau_slow = popt
        fitted = double_exponential(times, *popt)

        # Calculate fit statistics
        ss_res = np.sum((intensities - fitted)**2)
        ss_tot = np.sum((intensities - np.mean(intensities))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(intensities)
        k = 4  # number of parameters
        aic = n * np.log(ss_res/n + 1e-10) + 2 * k

        total_mobile = mf_fast + mf_slow

        # Use weighted average for effective tau
        if total_mobile > 0:
            effective_tau = (mf_fast * tau_fast + mf_slow * tau_slow) / total_mobile
        else:
            effective_tau = tau_fast

        return FRAPFitResult(
            model_type='double_exponential',
            mobile_fraction=min(total_mobile, 1.0),
            immobile_fraction=max(0, 1 - total_mobile),
            tau=effective_tau,
            t_half=effective_tau * np.log(2),
            r_squared=r_squared,
            aic=aic,
            fitted_curve=fitted,
            parameters={
                'mf_fast': mf_fast,
                'tau_fast': tau_fast,
                't_half_fast': tau_fast * np.log(2),
                'mf_slow': mf_slow,
                'tau_slow': tau_slow,
                't_half_slow': tau_slow * np.log(2)
            }
        )

    except Exception as e:
        warnings.warn(f"Double exponential fit failed: {e}")
        return None


# =============================================================================
# Main Analyzer Class
# =============================================================================

class FRAPAnalyzer:
    """
    Main class for FRAP analysis.

    Example usage:
        analyzer = FRAPAnalyzer()
        results = analyzer.analyze_lif('data.lif', series_pairs=[(9, 10)])
        analyzer.export_results(results, 'output/')
    """

    def __init__(self, auto_detect_rois: bool = True):
        """
        Initialize analyzer.

        Parameters:
            auto_detect_rois: Whether to automatically detect ROIs
        """
        self.auto_detect_rois = auto_detect_rois
        self.results: List[FRAPResult] = []

    def analyze_frames(self,
                       pre_frames: List[np.ndarray],
                       post_frames: List[np.ndarray],
                       time_interval: float,
                       name: str = "FRAP",
                       roi_data: Optional[ROIData] = None) -> FRAPResult:
        """
        Analyze FRAP data from image frames.

        Parameters:
            pre_frames: List of pre-bleach frames
            post_frames: List of post-bleach frames
            time_interval: Time between frames (seconds)
            name: Name/identifier for this dataset
            roi_data: Optional pre-computed ROI masks

        Returns:
            FRAPResult object with all analysis data
        """
        pre_bleach = pre_frames[-1].astype(float)

        # Detect or use provided ROIs
        if roi_data is None and self.auto_detect_rois:
            roi_data = detect_rois_auto(pre_bleach, post_frames[0])

        # Measure intensities
        pre_bleach_intensity = np.mean(pre_bleach[roi_data.bleach_mask])
        pre_reference_intensity = np.mean(pre_bleach[roi_data.reference_mask])
        background_intensity = np.mean(pre_bleach[roi_data.background_mask]) \
                              if np.any(roi_data.background_mask) else 0

        # Raw intensities
        raw_bleach = np.array([np.mean(f.astype(float)[roi_data.bleach_mask])
                              for f in post_frames])
        raw_reference = np.array([np.mean(f.astype(float)[roi_data.reference_mask])
                                 for f in post_frames])

        # Normalize
        double_norm = normalize_double(post_frames, roi_data,
                                       pre_bleach_intensity, pre_reference_intensity)
        full_scale_norm = normalize_full_scale(double_norm)

        # Time points
        times = np.array([t * time_interval for t in range(len(post_frames))])

        # Fit curves
        fit_single = fit_single_exponential(times, full_scale_norm)
        fit_double = fit_double_exponential(times, full_scale_norm)

        # Quality metrics
        quality_metrics = {
            'bleach_depth': 1 - (raw_bleach[0] / pre_bleach_intensity) if pre_bleach_intensity > 0 else 0,
            'recovery_extent': full_scale_norm[-1] if len(full_scale_norm) > 0 else 0,
            'reference_decay': 1 - (raw_reference[-1] / raw_reference[0]) if raw_reference[0] > 0 else 0
        }

        result = FRAPResult(
            name=name,
            times=times,
            raw_bleach_intensities=raw_bleach,
            raw_reference_intensities=raw_reference,
            background_intensity=background_intensity,
            double_normalized=double_norm,
            full_scale_normalized=full_scale_norm,
            fit_single=fit_single,
            fit_double=fit_double,
            roi_data=roi_data,
            time_interval=time_interval,
            quality_metrics=quality_metrics
        )

        self.results.append(result)
        return result

    def analyze_lif(self,
                    lif_path: str,
                    series_pairs: List[Tuple[int, int]],
                    names: Optional[List[str]] = None) -> List[FRAPResult]:
        """
        Analyze FRAP data from a Leica LIF file.

        Parameters:
            lif_path: Path to LIF file
            series_pairs: List of (pre_series_idx, post_series_idx) tuples
            names: Optional list of names for each dataset

        Returns:
            List of FRAPResult objects
        """
        if not HAS_READLIF:
            raise ImportError("readlif package required for LIF file support")

        lif = LifFile(lif_path)
        results = []

        for i, (pre_idx, post_idx) in enumerate(series_pairs):
            # Get series info
            pre_img = lif.get_image(pre_idx)
            post_img = lif.get_image(post_idx)

            # Extract time interval from metadata
            time_interval = pre_img.scale[3] if pre_img.scale[3] else 1.0

            # Load frames
            pre_frames = [np.array(f) for f in pre_img.get_iter_t(c=0, z=0)]
            post_frames = [np.array(f) for f in post_img.get_iter_t(c=0, z=0)]

            # Get name
            if names and i < len(names):
                name = names[i]
            else:
                name = lif.image_list[pre_idx]['name'].replace('/', '_').replace(' ', '_')

            result = self.analyze_frames(pre_frames, post_frames, time_interval, name)
            results.append(result)

        return results

    def analyze_tiff_stack(self,
                           tiff_path: str,
                           pre_frames_count: int,
                           time_interval: float,
                           name: str = "FRAP") -> FRAPResult:
        """
        Analyze FRAP data from a TIFF stack.

        Parameters:
            tiff_path: Path to TIFF stack
            pre_frames_count: Number of pre-bleach frames
            time_interval: Time between frames (seconds)
            name: Name for this dataset

        Returns:
            FRAPResult object
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for TIFF support")

        img = Image.open(tiff_path)
        frames = []

        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        pre_frames = frames[:pre_frames_count]
        post_frames = frames[pre_frames_count:]

        return self.analyze_frames(pre_frames, post_frames, time_interval, name)

    def export_results(self,
                       results: List[FRAPResult],
                       output_dir: str,
                       prefix: str = "FRAP") -> None:
        """
        Export results to CSV files.

        Parameters:
            results: List of FRAPResult objects
            output_dir: Output directory path
            prefix: Prefix for output filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Summary statistics
        summary_data = [r.summary_dict() for r in results]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / f"{prefix}_summary.csv", index=False)

        # Individual time courses
        for result in results:
            df = result.to_dataframe()
            safe_name = result.name.replace('/', '_').replace(' ', '_')
            df.to_csv(output_path / f"{prefix}_{safe_name}_timecourse.csv", index=False)

        # Combined time courses for plotting
        combined_data = []
        for result in results:
            for t, norm in zip(result.times, result.full_scale_normalized):
                combined_data.append({
                    'sample': result.name,
                    'time_s': t,
                    'normalized_intensity': norm
                })
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_path / f"{prefix}_all_timecourses.csv", index=False)

        print(f"Results exported to {output_path}/")

    def export_for_prism(self,
                         results: List[FRAPResult],
                         output_path: str) -> None:
        """
        Export data in GraphPad Prism-compatible format.

        Parameters:
            results: List of FRAPResult objects
            output_path: Output CSV path
        """
        # Find maximum number of time points
        max_times = max(len(r.times) for r in results)

        # Create wide-format DataFrame
        data = {'Time_s': []}
        for r in results:
            data[f'{r.name}_Normalized'] = []
            data[f'{r.name}_Fitted'] = []

        for i in range(max_times):
            # Use first result's times as reference
            if i < len(results[0].times):
                data['Time_s'].append(results[0].times[i])
            else:
                data['Time_s'].append(np.nan)

            for r in results:
                if i < len(r.full_scale_normalized):
                    data[f'{r.name}_Normalized'].append(r.full_scale_normalized[i])
                    if r.best_fit and r.best_fit.fitted_curve is not None:
                        data[f'{r.name}_Fitted'].append(r.best_fit.fitted_curve[i])
                    else:
                        data[f'{r.name}_Fitted'].append(np.nan)
                else:
                    data[f'{r.name}_Normalized'].append(np.nan)
                    data[f'{r.name}_Fitted'].append(np.nan)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Prism-compatible data exported to {output_path}")


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_recovery_curves(results: List[FRAPResult],
                         output_path: Optional[str] = None,
                         show_individual: bool = True,
                         show_fit: bool = True,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot FRAP recovery curves.

    Parameters:
        results: List of FRAPResult objects
        output_path: Path to save figure (optional)
        show_individual: Show individual data points
        show_fit: Show fitted curves
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for result, color in zip(results, colors):
        label = result.name

        if show_individual:
            ax.scatter(result.times, result.full_scale_normalized,
                      color=color, s=60, edgecolors='white', linewidth=1,
                      label=f'{label} (data)', zorder=5)

        if show_fit and result.best_fit and result.best_fit.fitted_curve is not None:
            # Plot smooth fit curve
            t_fine = np.linspace(0, result.times[-1] * 1.1, 100)
            if result.best_fit.model_type == 'single_exponential':
                y_fine = single_exponential(t_fine,
                                           result.best_fit.mobile_fraction,
                                           result.best_fit.tau)
            else:
                params = result.best_fit.parameters
                y_fine = double_exponential(t_fine,
                                           params['mf_fast'], params['tau_fast'],
                                           params['mf_slow'], params['tau_slow'])

            ax.plot(t_fine, y_fine, color=color, linewidth=2, alpha=0.7,
                   label=f'{label} (fit)')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Normalized Fluorescence', fontsize=12)
    ax.set_title('FRAP Recovery Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.close()


def plot_comparison_bars(results: List[FRAPResult],
                         output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 4)) -> None:
    """
    Plot bar charts comparing mobile fractions and half-times.

    Parameters:
        results: List of FRAPResult objects
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    names = [r.name for r in results]
    mobile_fractions = [r.best_fit.mobile_fraction * 100 if r.best_fit else 0 for r in results]
    half_times = [r.best_fit.t_half if r.best_fit else 0 for r in results]

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # Mobile fractions
    bars1 = ax1.bar(names, mobile_fractions, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Mobile Fraction (%)', fontsize=12)
    ax1.set_title('Mobile Fraction', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars1, mobile_fractions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Half-times
    bars2 = ax2.bar(names, half_times, color=colors, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Half-time (s)', fontsize=12)
    ax2.set_title('Recovery Half-time (t½)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars2, half_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.close()


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FRAP Analysis - Publication Standard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze LIF file with specific series
  python frap_analysis.py --input data.lif --series 9,10 --series 18,19 --output results/

  # Analyze TIFF stack
  python frap_analysis.py --input stack.tif --pre-frames 5 --interval 0.5 --output results/

  # List series in LIF file
  python frap_analysis.py --input data.lif --list-series
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input file (LIF or TIFF)')
    parser.add_argument('--output', '-o', default='frap_results/', help='Output directory')
    parser.add_argument('--series', '-s', action='append',
                        help='Series pair as "pre,post" (can specify multiple)')
    parser.add_argument('--names', '-n', action='append', help='Names for each series pair')
    parser.add_argument('--pre-frames', type=int, default=2,
                        help='Number of pre-bleach frames (TIFF only)')
    parser.add_argument('--interval', '-t', type=float, default=1.0,
                        help='Time interval between frames in seconds')
    parser.add_argument('--list-series', '-l', action='store_true',
                        help='List all series in LIF file and exit')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # List series mode
    if args.list_series:
        if input_path.suffix.lower() == '.lif':
            lif = LifFile(str(input_path))
            print(f"\nSeries in {input_path.name}:")
            print("-" * 50)
            for i, img in enumerate(lif.image_list):
                print(f"  {i:3d}: {img['name']}")
            print()
        else:
            print("--list-series only works with LIF files")
        return 0

    # Create analyzer
    analyzer = FRAPAnalyzer()

    # Analyze based on file type
    if input_path.suffix.lower() == '.lif':
        if not args.series:
            print("Error: Must specify --series for LIF files")
            return 1

        series_pairs = []
        for s in args.series:
            pre, post = map(int, s.split(','))
            series_pairs.append((pre, post))

        results = analyzer.analyze_lif(str(input_path), series_pairs, args.names)

    elif input_path.suffix.lower() in ['.tif', '.tiff']:
        results = [analyzer.analyze_tiff_stack(str(input_path),
                                               args.pre_frames,
                                               args.interval,
                                               input_path.stem)]
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        return 1

    # Export results
    output_path = Path(args.output)
    analyzer.export_results(results, str(output_path))
    analyzer.export_for_prism(results, str(output_path / 'FRAP_prism_format.csv'))

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        plot_recovery_curves(results, str(output_path / 'recovery_curves.png'))
        plot_comparison_bars(results, str(output_path / 'comparison_bars.png'))

    # Print summary
    print("\n" + "=" * 60)
    print("FRAP Analysis Summary")
    print("=" * 60)
    for r in results:
        print(f"\n{r.name}:")
        if r.best_fit:
            print(f"  Mobile Fraction: {r.best_fit.mobile_fraction*100:.1f}%")
            print(f"  Half-time (t½): {r.best_fit.t_half:.3f} s")
            print(f"  Time constant (τ): {r.best_fit.tau:.3f} s")
            print(f"  R²: {r.best_fit.r_squared:.4f}")
            print(f"  Model: {r.best_fit.model_type}")

    return 0


if __name__ == '__main__':
    exit(main())
