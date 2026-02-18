import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import NavigationToolbar2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import pearsonr

class JunctionAnalyzer:

    def __init__(self, pixel_size_m):
        self.pixel_size_m = pixel_size_m  # meters per pixel
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Aplica configuraciones globales para que las figuras se vean profesionales y restringe la toolbar."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'lines.linewidth': 2,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#cccccc'
        })

        # --- RESTRICCIÓN DE LA BARRA DE HERRAMIENTAS ---
        NavigationToolbar2.toolitems = (
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
        )

    def detect(self, roi, manual_line, roi_current=None, weight_current=10.0, 
               plot_a=False, plot_b=False, plot_c=False, sweep_weights=None, _sweep_call=False):
        h, w = roi.shape

        # --- resample manual_line to match ROI width ---
        if len(manual_line) != w:
            t_manual = np.linspace(0.0, 1.0, len(manual_line))
            t_target = np.linspace(0.0, 1.0, w)
            fx = interp1d(t_manual, manual_line[:, 0], kind='linear', fill_value='extrapolate')
            fy = interp1d(t_manual, manual_line[:, 1], kind='linear', fill_value='extrapolate')
            manual_line_rs = np.column_stack([fx(t_target), fy(t_target)])
        else:
            manual_line_rs = np.array(manual_line, dtype=float)

        # -----------------------------------------------------------------
        # Plot (A): SEM ROI - EBIC / Current ROI
        # -----------------------------------------------------------------
        if plot_a:
            try:
                if roi_current is not None:
                    # Orientación dinámica basada en la relación de aspecto de la ROI
                    if w > h:
                        fig, axes = plt.subplots(2, 1, figsize=(10, 8)) # Uno encima de otro
                    else:
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # Uno al lado del otro
                    
                    # SEM
                    im0 = axes[0].imshow(roi, cmap='gray', origin='upper', aspect='equal')
                    axes[0].set_title('SEM ROI (Topography)')
                    axes[0].set_xlabel('Width (pixels)')
                    axes[0].set_ylabel('Height (pixels)')
                    div0 = make_axes_locatable(axes[0])
                    cax0 = div0.append_axes("right", size="3%", pad=0.1)
                    fig.colorbar(im0, cax=cax0).set_label('Intensity')
                    
                    # EBIC
                    im1 = axes[1].imshow(roi_current, cmap='viridis', origin='upper', aspect='equal')
                    axes[1].set_title('EBIC / Current ROI')
                    axes[1].set_xlabel('Width (pixels)')
                    axes[1].set_ylabel('Height (pixels)')
                    div1 = make_axes_locatable(axes[1])
                    cax1 = div1.append_axes("right", size="3%", pad=0.1)
                    fig.colorbar(im1, cax=cax1).set_label('Current (nA)')
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                    im = ax.imshow(roi, cmap='gray', origin='upper', aspect='equal')
                    ax.set_title('SEM ROI (Topography)')
                    ax.set_xlabel('Width (pixels)')
                    ax.set_ylabel('Height (pixels)')
                    div = make_axes_locatable(ax)
                    cax = div.append_axes("right", size="3%", pad=0.1)
                    fig.colorbar(im, cax=cax).set_label('Intensity')
                
                fig.suptitle("Region of Interest (ROI) Extraction (1:1 Aspect Ratio)", fontsize=15, y=0.98)
                fig.tight_layout()
                fig.show() 
            except Exception as e: print(e)

        results = []

        # --- Method : Canny with Bilateral pre-filtering, with post-processing ---
        try:
            filtered_roi = self._apply_preprocessing_filter(roi)
            filtered_current = None
            if roi_current is not None:
                try:
                    filtered_current = self._apply_preprocessing_filter(roi_current)
                except Exception:
                    filtered_current = roi_current.astype(np.uint8)

            detected_roi_coords = self._detect_junction_canny(filtered_roi, roi_current=filtered_current, weight_current=weight_current)
            
            if detected_roi_coords.shape[0] != w:
                t_det = np.linspace(0.0, 1.0, detected_roi_coords.shape[0])
                t_new = np.linspace(0.0, 1.0, w)
                fcx = interp1d(t_det, detected_roi_coords[:, 0], kind='linear', fill_value='extrapolate')
                fcy = interp1d(t_det, detected_roi_coords[:, 1], kind='linear', fill_value='extrapolate')
                detected_roi_coords = np.column_stack([fcx(t_new), fcy(t_new)])

            postprocessed_roi_coords = self._fit_line_postprocessing(detected_roi_coords)

            detected_image_coords = self._map_detected_to_image_coords(manual_line_rs, postprocessed_roi_coords, roi_height=h)
            metrics = self._compare_with_manual(manual_line_rs, detected_image_coords)
            
            # -----------------------------------------------------------------
            # Plot (B): Junction Detection Comparison
            # -----------------------------------------------------------------
            if plot_b:
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                    ax.imshow(filtered_roi, cmap='gray', origin='upper', aspect='equal')
                    
                    manual_roi_y = (h - 1) / 2.0
                    ax.axhline(y=manual_roi_y, color='#00FF00', linestyle='--', linewidth=2, label='Manual reference (Center)', alpha=0.8)
                    
                    if filtered_current is not None:
                        detected_sem_only = self._detect_junction_canny(filtered_roi, roi_current=filtered_current, weight_current=0.0)
                        postproc_sem = self._fit_line_postprocessing(detected_sem_only)
                        ax.plot(postproc_sem[:, 0], postproc_sem[:, 1], color='#1f77b4', linestyle='-', linewidth=2, label='SEM only (w=0)', alpha=0.9)
                        
                        detected_ebic_only = self._detect_junction_canny(filtered_roi, roi_current=filtered_current, weight_current=1e6)
                        postproc_ebic = self._fit_line_postprocessing(detected_ebic_only)
                        ax.plot(postproc_ebic[:, 0], postproc_ebic[:, 1], color='#ff7f0e', linestyle='-', linewidth=2, label='EBIC only (w=1e6)', alpha=0.9)
                    
                    ax.plot(detected_roi_coords[:, 0], detected_roi_coords[:, 1], 'y.', markersize=4, label=f'Raw Detections (w={weight_current})', alpha=0.7)
                    ax.plot(postprocessed_roi_coords[:, 0], postprocessed_roi_coords[:, 1], color='#d62728', linestyle='-', linewidth=2.5, label=f'Combined Fit (w={weight_current})')
                    
                    try:
                        roi8 = filtered_roi if filtered_roi.dtype == np.uint8 else cv2.normalize(filtered_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        otsu_r, _ = cv2.threshold(roi8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        edges_sem = cv2.Canny(roi8, 0.5 * float(otsu_r), float(otsu_r))
                        ys_sem, xs_sem = np.where(edges_sem > 0)
                        if ys_sem.size > 0: ax.scatter(xs_sem, ys_sem, s=2, c='magenta', label='SEM Edges', alpha=0.4)
                    except Exception: pass
                    
                    if filtered_current is not None:
                        try:
                            roi_curr8 = filtered_current if filtered_current.dtype == np.uint8 else cv2.normalize(filtered_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            otsu_c, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            edges_curr = cv2.Canny(roi_curr8, 0.5 * float(otsu_c), float(otsu_c))
                            ys_c, xs_c = np.where(edges_curr > 0)
                            if ys_c.size > 0: ax.scatter(xs_c, ys_c, s=2, c='cyan', label='EBIC Edges', alpha=0.6)
                        except Exception: pass

                    ax.set_title(f'Junction Detection Profile Analysis (1:1 Aspect Ratio)', pad=15)
                    ax.set_xlabel('ROI Width (pixels)')
                    ax.set_ylabel('ROI Height (pixels)')
                    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
                    fig.tight_layout()
                    fig.show()
                except Exception as e: print(e)

            # -----------------------------------------------------------------
            # Plot (C): Raw vs Filtered EBIC
            # -----------------------------------------------------------------
            if plot_c and roi_current is not None:
                try:
                    # Orientación dinámica basada en la relación de aspecto de la ROI
                    if w > h:
                        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8)) # Uno encima de otro
                    else:
                        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5)) # Uno al lado del otro
                    
                    im0 = axes2[0].imshow(roi_current, cmap='viridis', origin='upper', aspect='equal')
                    axes2[0].set_title('Raw EBIC ROI')
                    axes2[0].set_xlabel('Width (pixels)')
                    axes2[0].set_ylabel('Height (pixels)')
                    div0 = make_axes_locatable(axes2[0])
                    cax0 = div0.append_axes("right", size="3%", pad=0.1)
                    fig2.colorbar(im0, cax=cax0).set_label('Current (nA)')

                    try:
                        roi_curr8_raw = roi_current if roi_current.dtype == np.uint8 else cv2.normalize(roi_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        otsu_cr, _ = cv2.threshold(roi_curr8_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        edges_curr_raw = cv2.Canny(roi_curr8_raw, 0.5 * float(otsu_cr), float(otsu_cr))
                        ys_c, xs_c = np.where(edges_curr_raw > 0)
                        if ys_c.size > 0: axes2[0].scatter(xs_c, ys_c, s=1.5, c='magenta', label='Canny Edges', alpha=0.7)
                        axes2[0].legend(loc='lower right', fontsize='small')
                    except Exception: pass

                    if filtered_current is not None:
                        im1 = axes2[1].imshow(filtered_current, cmap='viridis', origin='upper', aspect='equal')
                        axes2[1].set_title('Bilateral Filtered EBIC ROI')
                        axes2[1].set_xlabel('Width (pixels)')
                        axes2[1].set_ylabel('Height (pixels)')
                        div1 = make_axes_locatable(axes2[1])
                        cax1 = div1.append_axes("right", size="3%", pad=0.1)
                        fig2.colorbar(im1, cax=cax1).set_label('Current (nA)')
                        
                        try:
                            roi_curr8 = filtered_current if filtered_current.dtype == np.uint8 else cv2.normalize(filtered_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            otsu_cc, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            edges_curr = cv2.Canny(roi_curr8, 0.5 * float(otsu_cc), float(otsu_cc))
                            ys2, xs2 = np.where(edges_curr > 0)
                            if ys2.size > 0: axes2[1].scatter(xs2, ys2, s=1.5, c='magenta', label='Canny Edges', alpha=0.7)
                            axes2[1].legend(loc='lower right', fontsize='small')
                        except Exception: pass
                        
                    fig2.suptitle("Filtering Effect on EBIC Signal (1:1 Aspect Ratio)", fontsize=15, y=0.98)
                    fig2.tight_layout()
                    fig2.show()
                except Exception as e: print(e)

            results.append(("Canny (Filtered, Spline)", detected_image_coords, metrics))
        except Exception as e:
            print(f"[Canny (Filtered, Spline)] failed: {e}")

        return results

    def _apply_preprocessing_filter(self, roi):
        normalized_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.bilateralFilter(normalized_roi, d=9, sigmaColor=75, sigmaSpace=9)

    def _apply_spline_postprocessing(self, detected_coords):
        t = np.linspace(0, 1, len(detected_coords))
        tck_x = splrep(t, detected_coords[:, 0], s=0.5) 
        tck_y = splrep(t, detected_coords[:, 1], s=0.5)
        t_smooth = np.linspace(0, 1, 1000)
        x_smooth = splev(t_smooth, tck_x)
        y_smooth = splev(t_smooth, tck_y)
        return np.column_stack([x_smooth, y_smooth])

    def compute_gradient_stats(self, roi, roi_current):
        sem = roi.astype(float)
        cur = roi_current.astype(float)
        h_sem, w_sem = sem.shape
        h_cur, w_cur = cur.shape
        h = min(h_sem, h_cur)
        w = min(w_sem, w_cur)
        if (h, w) != (h_sem, w_sem): sem = sem[:h, :w]
        if (h, w) != (h_cur, w_cur): cur = cur[:h, :w]
        sem_grads = np.abs(np.gradient(sem, axis=0))
        cur_grads = np.abs(np.gradient(cur, axis=0))
        sem_stats = {"max": float(np.max(sem_grads)), "mean": float(np.mean(sem_grads)), "median": float(np.median(sem_grads))}
        curr_stats = {"max": float(np.max(cur_grads)), "mean": float(np.mean(cur_grads)), "median": float(np.median(cur_grads))}
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = np.where(sem_grads != 0, cur_grads / (sem_grads + 1e-12), 0.0)
        ratios = {"max_ratio": float(np.nanmax(ratio_arr)), "mean_ratio": float(np.nanmean(ratio_arr))}
        return sem_stats, curr_stats, ratios
    
    def _detect_junction_canny(self, roi, roi_current=None, weight_current=10.0, debug=False):
        H, W = roi.shape
        detected = np.zeros((W, 2), dtype=float)
        roi_8bit = roi.astype(np.uint8)
        otsu_val, _ = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh = float(otsu_val)
        low_thresh = 0.5 * high_thresh
        edges = cv2.Canny(roi_8bit, low_thresh, high_thresh)

        curr_edges = None
        if roi_current is not None:
            try:
                roi_curr_8 = roi_current.astype(np.uint8)
                otsu_cur, _ = cv2.threshold(roi_curr_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                high_cur = float(otsu_cur)
                low_cur = 0.5 * high_cur
                curr_edges = cv2.Canny(roi_curr_8, low_cur, high_cur)
            except Exception: curr_edges = None

        if roi_current is not None:
            has_current = True
            roi_current_f = roi_current.astype(float)
        else:
            has_current = False
            roi_current_f = None

        for col in range(W):
            edge_rows_sem = np.where(edges[:, col] > 0)[0]
            if curr_edges is not None: edge_rows_curr = np.where(curr_edges[:, col] > 0)[0]
            else: edge_rows_curr = np.array([], dtype=int)

            if edge_rows_sem.size == 0 and edge_rows_curr.size == 0: edge_rows = np.array([], dtype=int)
            else: edge_rows = np.unique(np.concatenate([edge_rows_sem, edge_rows_curr]))
            
            profile = roi[:, col].astype(float)
            grad = np.gradient(profile)

            curr_grad = None
            if has_current:
                curr_profile = roi_current_f[:, col]
                curr_grad = np.gradient(curr_profile)

            row_idx = 0
            eps = 1e-12

            if edge_rows.size > 0:
                if curr_grad is None:
                    sem_edge = np.abs(grad[edge_rows])
                    sem_norm = sem_edge / (np.max(sem_edge) + eps)
                    max_grad_idx = edge_rows[np.argmax(sem_norm)]
                    row_idx = int(max_grad_idx)
                else:
                    sem_scores = np.abs(grad[edge_rows])
                    curr_scores = np.abs(curr_grad[edge_rows])
                    sem_norm = sem_scores / (np.max(sem_scores) + eps)
                    curr_norm = curr_scores / (np.max(curr_scores) + eps)
                    neigh = 3
                    candidates = []
                    for r in edge_rows:
                        r0 = max(0, r - neigh)
                        r1 = min(H - 1, r + neigh)
                        candidates.extend(range(r0, r1 + 1))
                    candidates = np.unique(candidates)
                    sem_col = np.abs(grad[candidates])
                    curr_col = np.abs(curr_grad[candidates])
                    sem_col_norm = sem_col / (np.max(sem_col) + eps)
                    curr_col_norm = curr_col / (np.max(curr_col) + eps)
                    comb_col = sem_col_norm + weight_current * curr_col_norm
                    best_idx = int(candidates[np.argmax(comb_col)])
                    row_idx = best_idx
            else:
                if curr_grad is None:
                    col_abs = np.abs(grad)
                    col_norm = col_abs / (np.max(col_abs) + eps)
                    row_idx = int(np.argmax(col_norm))
                else:
                    sem_col = np.abs(grad)
                    curr_col = np.abs(curr_grad)
                    sem_col_norm = sem_col / (np.max(sem_col) + eps)
                    curr_col_norm = curr_col / (np.max(curr_col) + eps)
                    comb = sem_col_norm + weight_current * curr_col_norm
                    row_idx = int(np.argmax(comb))
            detected[col] = [col, row_idx]

        return detected

    def _map_detected_to_image_coords(self, manual_line_rs, detected_roi_coords, roi_height):
        W = len(manual_line_rs)
        H = roi_height
        half = (H - 1) / 2.0
        image_coords = np.zeros((W, 2), dtype=float)
        tangents = np.zeros_like(manual_line_rs)
        tangents[1:-1] = (manual_line_rs[2:] - manual_line_rs[:-2]) / 2.0
        tangents[0] = manual_line_rs[1] - manual_line_rs[0]
        tangents[-1] = manual_line_rs[-1] - manual_line_rs[-2]
        perp_vectors = np.array([-tangents[:, 1], tangents[:, 0]]).T
        perp_norms = np.linalg.norm(perp_vectors, axis=1)
        perp_units = np.zeros_like(perp_vectors)
        non_zero = perp_norms != 0
        perp_units[non_zero] = perp_vectors[non_zero] / perp_norms[non_zero][:, np.newaxis]
        for i in range(W):
            cx, cy = manual_line_rs[i]
            row_idx = detected_roi_coords[i, 1]
            offset = float(row_idx) - half
            img_x = cx + perp_units[i, 0] * offset
            img_y = cy + perp_units[i, 1] * offset
            image_coords[i] = [img_x, img_y]
        return image_coords

    def _compare_with_manual(self, manual_line, detected_line):
        if manual_line.shape != detected_line.shape:
            t_manual = np.linspace(0, 1, len(manual_line))
            t_detect = np.linspace(0, 1, len(detected_line))
            f_x = interp1d(t_detect, detected_line[:, 0], kind='linear', fill_value="extrapolate")
            f_y = interp1d(t_detect, detected_line[:, 1], kind='linear', fill_value="extrapolate")
            detected_line = np.column_stack([f_x(t_manual), f_y(t_manual)])
        diffs = np.linalg.norm(detected_line - manual_line, axis=1) * self.pixel_size_m * 1e6
        mean_dev = np.mean(diffs)
        std_dev = np.std(diffs)
        max_dev = np.max(diffs)
        corr_x, _ = pearsonr(manual_line[:, 0], detected_line[:, 0])
        return mean_dev, std_dev, max_dev

    def visualize_results(self, image, manual_line, results):
        """ Plot (D): General Result Over Main Image """
        for name, line_imgcoords, metrics in results:
            if len(metrics) == 4:
                mean_dev, std_dev, max_dev, r2 = metrics
            else:
                mean_dev, std_dev, max_dev = metrics
                r2 = float('nan')

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(image, cmap='gray', origin='upper', aspect='equal')
            
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im, cax=cax).set_label('Intensity')
            
            ax.plot(manual_line[:, 0], manual_line[:, 1], color='#ff7f0e', linestyle='--', linewidth=2, label='Manual Estimation')
            ax.plot(line_imgcoords[:, 0], line_imgcoords[:, 1], color='#00FF00', linestyle='-', linewidth=2.5, label='Detected Junction')
            
            title = (f"General Junction Detection ({name})\n"
                     f"Mean Deviation: {mean_dev:.2f} \u03BCm | Std Dev: {std_dev:.2f} \u03BCm | Max Dev: {max_dev:.2f} \u03BCm")
            
            ax.set_title(title, pad=15)
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            ax.legend(loc='best')
            fig.tight_layout()
            
            fig.show() 

    def _fit_line_postprocessing(self, detected_coords):
        x = detected_coords[:, 0]
        y = detected_coords[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        x_fit = np.linspace(x.min(), x.max(), len(x))
        y_fit = a * x_fit + b
        return np.column_stack([x_fit, y_fit])