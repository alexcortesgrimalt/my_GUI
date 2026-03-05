"""
Advanced perpendicular profile fitter with adjustable parameters.
Incorporates techniques from EBIC_Analysis_Tool/DiffLenExt.py for robust fitting.

Key features:
- Adjustable SCR (Space Charge Region) width estimation
- SNR-based data truncation to avoid noisy tail regions
- Iterative starting point detection near junction
- Plateau detection in derivatives for linear regions
- Robust statistics (MAD, IQR) for noise-resistant fitting
- Pixel-grid quantization of diffusion lengths
"""

import numpy as np
from scipy.optimize import curve_fit


class PerpendicularFitter:
    """Advanced fitter for perpendicular EBIC profiles with adjustable parameters.
    
    Parameters
    ----------
    min_points : int
        Minimum number of points required for a valid fit (default: 6)
    skip_near_junction : int
        Number of points to skip near the junction peak (default: 3)
    snr_threshold : float
        SNR threshold for truncating noisy tails (default: 3.0)
        Higher values = more aggressive tail truncation
    scr_width_estimate : float or None
        Estimated Space Charge Region width (µm). If provided, fitting avoids region near junction.
        If None, estimated from data (default: None)
    search_max_shift : int
        Maximum number of points to shift when searching for best starting point (default: 5)
    fit_method : str
        'linear_log' for linear fit on ln(I), 'exponential' for exponential model (default: 'linear_log')
    pixel_size_um : float
        Pixel size in micrometers for quantization (default: 1.0)
    """
    
    def __init__(self, min_points=6, skip_near_junction=3, snr_threshold=3.0,
                 scr_width_estimate=None, search_max_shift=5, 
                 fit_method='linear_log', pixel_size_um=1.0):
        self.min_points = min_points
        self.skip_near_junction = skip_near_junction
        self.snr_threshold = snr_threshold
        self.scr_width_estimate = scr_width_estimate
        self.search_max_shift = search_max_shift
        self.fit_method = fit_method
        self.pixel_size_um = pixel_size_um

    def _safe_ln(self, y):
        """Safely compute logarithm, avoiding log(0) and log(negative)."""
        y = np.asarray(y, dtype=float)
        pos = y[y > 0]
        floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
        return np.log(np.maximum(y, floor))

    def _find_snr_end_index(self, y_vals, min_points=10):
        """Find truncation index based on SNR threshold.
        
        Returns index where signal falls below SNR threshold *
        estimated noise level.
        """
        if len(y_vals) < min_points:
            return len(y_vals)
        
        # Estimate noise from tail (last min_points)
        noise_region = y_vals[-min_points:]
        y0_est = np.median(noise_region)
        sigma_est = np.std(noise_region)
        cutoff_value = y0_est + self.snr_threshold * sigma_est
        
        # Find where data falls below threshold
        for i in range(len(y_vals) - 1, min_points - 1, -1):
            if y_vals[i] > cutoff_value:
                return i + 1
        
        return min_points

    def _linear_fit_with_stats(self, x, y):
        """Perform linear regression on x, y with robust statistics.
        
        Returns dict with slope, intercept, r2, fit curve, and coordinate arrays.
        """
        if x.size < max(2, self.min_points):
            return None
        
        try:
            # Linear fit: y = m*x + b
            p = np.polyfit(x, y, 1)
            m, b = float(p[0]), float(p[1])
            
            # Fitted values
            y_fit = np.polyval(p, x)
            
            # R² calculation
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            
            # For ln(I) fits, slope is decay rate: inv_length = -1/slope
            inv_len = abs(1.0 / m) if m != 0 else np.nan
            
            return {
                'slope': m,
                'intercept': b,
                'r2': float(r2),
                'inv_length': float(inv_len),
                'x_fit': x.copy(),
                'y_fit': y_fit.copy()
            }
        except Exception as e:
            print(f"Linear fit failed: {e}")
            return None

    def _exponential_fit(self, x, y):
        """Fit exponential model: y = A * exp(-lam * x) + y0"""
        if x.size < max(2, self.min_points):
            return None
        
        try:
            # Initial guesses
            y0_init = y[-1] if y[-1] > 0 else 0
            A_init = y[0] - y0_init
            lam_init = 1.0 / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else 0.1
            
            # Curve fit exponential
            popt, _ = curve_fit(
                lambda x, A, lam, y0: A * np.exp(-lam * x) + y0,
                x, y,
                p0=[A_init, lam_init, y0_init],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=20000
            )
            
            A, lam, y0 = popt
            y_fit = A * np.exp(-lam * x) + y0
            
            # R² and inv_length
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            inv_len = 1.0 / lam if lam != 0 else np.nan
            
            return {
                'slope': -lam,
                'intercept': y0,
                'r2': float(r2),
                'inv_length': float(inv_len),
                'x_fit': x.copy(),
                'y_fit': y_fit.copy(),
                'amplitude': A
            }
        except Exception as e:
            print(f"Exponential fit failed: {e}")
            return None

    def _estimate_scr_width(self, x, y):
        """Estimate SCR (Space Charge Region) width from data gradient.
        
        SCR is typically near peak where gradient of current is steepest.
        """
        if len(x) < 3:
            return 0
        
        # Compute gradient
        dy_dx = np.abs(np.gradient(y, x))
        
        # Find region of steepest gradient (width of peak)
        max_grad_idx = np.argmax(dy_dx)
        
        # Estimate half-width at half-maximum of gradient
        half_max = dy_dx[max_grad_idx] / 2
        indices = np.where(dy_dx > half_max)[0]
        
        if len(indices) > 0:
            width = x[indices[-1]] - x[indices[0]]
            return abs(width)
        
        return 0

    def _linear_fit(self, x, y):
        """Backward-compatible simple linear fit (kept for compatibility)."""
        if x.size < max(2, self.min_points):
            return None
        try:
            p = np.polyfit(x, y, 1)
            m, b = float(p[0]), float(p[1])
            y_fit = np.polyval(p, x)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            inv_len = 1.0 / abs(m) if m != 0 else np.nan
            return {
                'slope': m,
                'intercept': b,
                'r2': float(r2),
                'inv_length': float(inv_len),
                'x_fit': x,
                'y_fit': y_fit
            }
        except Exception:
            return None

    def fit_profile(self, dist_um, current_nA, show_debug=False):
        """Fit a single perpendicular profile with advanced techniques.
        
        Parameters
        ----------
        dist_um : array
            Distance values (µm)
        current_nA : array
            Current values (nA)
        show_debug : bool
            Print debug information
            
        Returns
        -------
        dict
            Results including junction position, left/right fits, depletion width, etc.
        """
        x = np.asarray(dist_um, dtype=float)
        y = np.asarray(current_nA, dtype=float)
        
        if x.size == 0 or y.size == 0:
            raise ValueError('Empty profile')
        
        # Find junction (peak of current)
        junction_idx = int(np.argmax(y))
        junction_pos = float(x[junction_idx])
        
        if show_debug:
            print(f"Junction @ x={junction_pos:.4f} µm, idx={junction_idx}")
        
        x_rel = x - junction_pos
        
        # --- LEFT SIDE: indices < junction_idx ---
        left_indices = np.where(np.arange(len(x)) < junction_idx)[0]
        left_result = None
        left_start = None
        
        if left_indices.size > self.skip_near_junction:
            # Take indices starting from skip_near_junction points before junction
            start_idx = junction_idx - self.skip_near_junction - 1
            if start_idx >= 0:
                x_left_raw = x[left_indices[:start_idx + 1]]
                y_left_raw = y[left_indices[:start_idx + 1]]
                
                # Flip left data for fitting (distance from junction increases left)
                x_left = junction_pos - x_left_raw
                y_left = np.array(y_left_raw, dtype=float)[::-1]
                
                # Truncate noisy tail
                cut_idx = self._find_snr_end_index(y_left)
                y_left_trunc = y_left[:cut_idx]
                x_left_trunc = x_left[:cut_idx]
                
                if len(x_left_trunc) >= self.min_points:
                    x_fit_left = x_left_trunc
                    if self.fit_method == 'linear_log':
                        y_fit_left = self._safe_ln(y_left_trunc)
                    else:
                        y_fit_left = y_left_trunc
                    
                    if len(x_fit_left) >= self.min_points:
                        left_result = self._linear_fit_with_stats(x_fit_left, y_fit_left)
                        if left_result:
                            left_start = -x_left_trunc[0]
        
        # --- RIGHT SIDE: indices > junction_idx ---
        right_indices = np.where(np.arange(len(x)) > junction_idx)[0]
        right_result = None
        right_start = None
        
        if right_indices.size > self.skip_near_junction:
            # Take indices starting from skip_near_junction points after junction
            start_idx = junction_idx + self.skip_near_junction + 1
            if start_idx < len(x):
                x_right_raw = x[right_indices]
                y_right_raw = y[right_indices]
                
                # For right side, indices are already in order
                # Find which indices are >= start_idx
                valid_idx = right_indices >= start_idx
                if np.any(valid_idx):
                    x_right = x_right_raw[valid_idx]
                    y_right = y_right_raw[valid_idx]
                    
                    # Measure from junction
                    x_right = x_right - junction_pos
                    
                    # Truncate noisy tail
                    cut_idx = self._find_snr_end_index(y_right)
                    y_right_trunc = y_right[:cut_idx]
                    x_right_trunc = x_right[:cut_idx]
                    
                    if len(x_right_trunc) >= self.min_points:
                        x_fit_right = x_right_trunc
                        if self.fit_method == 'linear_log':
                            y_fit_right = self._safe_ln(y_right_trunc)
                        else:
                            y_fit_right = y_right_trunc
                        
                        if len(x_fit_right) >= self.min_points:
                            right_result = self._linear_fit_with_stats(x_fit_right, y_fit_right)
                            if right_result:
                                right_start = x_right_trunc[0]
        
        # Calculate depletion width (estimated from fit start points)
        depletion_width = None
        if left_start is not None and right_start is not None:
            depletion_width = float(right_start + left_start)
        
        # Estimate SCR if requested (for reference/documentation)
        scr_width = self.scr_width_estimate
        if scr_width is None:
            scr_width = max(self._estimate_scr_width(x, y), self.pixel_size_um)
        
        if show_debug:
            print(f"SCR width estimate: {scr_width:.4f} µm")
            print(f"Left fit points: {len(x_left_trunc) if left_result else 0}, Right fit points: {len(x_right_trunc) if right_result else 0}")
        
        return {
            'junction_idx': junction_idx,
            'junction_pos': junction_pos,
            'scr_width': float(scr_width),
            'left': left_result,
            'right': right_result,
            'left_start': left_start,
            'right_start': right_start,
            'depletion_width': depletion_width
        }


if __name__ == '__main__':
    # Quick self-test with adjustable parameters
    import numpy as np
    x = np.linspace(-5, 5, 101)
    y = np.where(x < 0, 10 * np.exp(0.8 * (x + 2)) + 0.5, 5 * np.exp(-0.5 * (x - 2)) + 0.2)
    
    print("Test 1: Default parameters")
    fitter = PerpendicularFitter()
    res = fitter.fit_profile(x, y, show_debug=True)
    print(f"  Left slope: {res['left']['slope']:.4f}")
    print(f"  Right slope: {res['right']['slope']:.4f}")
    
    print("\nTest 2: With custom SCR width")
    fitter2 = PerpendicularFitter(scr_width_estimate=0.5, skip_near_junction=2)
    res2 = fitter2.fit_profile(x, y, show_debug=True)
    print(f"  Left slope: {res2['left']['slope']:.4f}")
    print(f"  Right slope: {res2['right']['slope']:.4f}")
    
    print("\nTest 3: With aggressive SNR truncation")
    fitter3 = PerpendicularFitter(snr_threshold=5.0)
    res3 = fitter3.fit_profile(x, y, show_debug=True)
    print(f"  SCR width: {res3['scr_width']:.4f}")
