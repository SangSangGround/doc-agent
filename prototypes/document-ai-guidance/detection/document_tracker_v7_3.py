"""Close-range *experimental* ORB document alignment for v7.3.

Whole-document tracker is first choice. For a hand-held phone with only part
of the original paper visible, match textured interior regions of the known
reference photo to the live frame. The projected A4 corners may be outside the
frame. Accept only dispersed, geometrically consistent matches, require two
stable frames, and NEVER return the prior homography after losing tracking.

This does not guarantee tracking with extreme zoom, severe blur, repeated form
boxes or finger occlusion. Never use as a proof of pen/finger contact.
"""
from __future__ import annotations
import math
import cv2
import numpy as np

from document_tracker_v7 import PaperTracker, TrackResult, _preprocess, _warp, resize_limit


class CloseRangePaperTracker(PaperTracker):
    def __init__(self, manifest, reference_path, max_dim=1080):
        super().__init__(manifest, reference_path, allow_contour=False,
                         max_dim=max_dim, min_inliers=18, min_inlier_ratio=.39,
                         max_jump_fraction=.20)
        self.close_counter=0

    def orb_match(self, frame):
        # Retain strict whole-paper matching when the sheet is in view.
        if self.last_good is None or self.last_good.method == 'ORB':
            whole, reason = super().orb_match(frame)
            if whole is not None:
                return whole, ''
        h, w = frame.shape[:2]
        small, to_small = resize_limit(frame, self.max_dim)
        sh, sw = small.shape[:2]
        gray = _preprocess(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 35:
            return None, 'close mode: too few live features'
        pairs = self.matcher.knnMatch(self.desc_ref, desc, k=2)
        matches = [a for pair in pairs if len(pair) == 2
                   for a, b in [pair] if a.distance < .72 * b.distance]
        if len(matches) < 28:
            return None, f'close mode: only {len(matches)} good matches'
        ref_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in matches])
        live_pts = np.float32([kps[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(ref_pts, live_pts, cv2.RANSAC, 3.5,
                                    maxIters=3000, confidence=.995)
        if H is None or mask is None:
            return None, 'close mode: RANSAC failed'
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        ratio = count / len(matches)
        if count < 24 or ratio < .53:
            return None, f'close mode: weak inliers {count}/{len(matches)}'
        r, l = ref_pts[inliers], live_pts[inliers]
        # Reject matches to a thin line, a single table cell or repeated print.
        ref_quad = _warp(self.ref_corners, self.ref_to_small)
        rx = float(np.ptp(r[:, 0])) / max(1, float(np.ptp(ref_quad[:, 0])))
        ry = float(np.ptp(r[:, 1])) / max(1, float(np.ptp(ref_quad[:, 1])))
        lx = float(np.ptp(l[:, 0])) / sw
        ly = float(np.ptp(l[:, 1])) / sh
        hull_area = cv2.contourArea(cv2.convexHull(l.reshape(-1, 1, 2))) / (sw * sh)
        if rx < .12 or ry < .11 or lx < .22 or ly < .17 or hull_area < .023:
            return None, f'close mode: narrow match spread ref={rx:.2f},{ry:.2f} live={lx:.2f},{ly:.2f}'
        try:
            Hinv = np.linalg.inv(H)
            error_forward = np.linalg.norm(_warp(r, H) - l, axis=1)
            error_back = np.linalg.norm(_warp(l, Hinv) - r, axis=1)
            err = float(np.median(error_forward))
            if err > 2.7 or float(np.median(error_back)) > 4.0:
                return None, f'close mode: reprojection error {err:.1f}px'
            H_ref_live = np.linalg.inv(to_small) @ H @ self.ref_to_small
            corners = _warp(self.ref_corners, H_ref_live)
            if not np.isfinite(corners).all() or not cv2.isContourConvex(np.float32(corners)):
                return None, 'close mode: invalid projected corners'
            if cv2.contourArea(np.float32(corners)) <= .06 * w * h:
                return None, 'close mode: projected paper implausibly small'
            if np.max(np.abs(corners)) > 15 * max(w, h):
                return None, 'close mode: unstable projected scale'
            # Reject significant flips/mirrors. Same paper orientation as reference.
            if cv2.contourArea(np.float32(corners), oriented=True) <= 0:
                return None, 'close mode: inverted page geometry'
            H_live_mm = self.H_ref_to_mm @ np.linalg.inv(H_ref_live)
            if not np.isfinite(H_live_mm).all() or abs(np.linalg.det(H_live_mm)) < 1e-14:
                return None, 'close mode: degenerate coordinates'
            # Require that at least some part of live frame corresponds to the A4.
            probes = np.float32([[w/2,h/2], [w*.25,h*.25], [w*.75,h*.75],
                                 [w*.25,h*.75], [w*.75,h*.25]])
            mapped = _warp(probes, H_live_mm)
            paper_w = float(self.manifest['paper_width_mm'])
            paper_h = float(self.manifest['paper_height_mm'])
            inside = ((mapped[:,0]>-12)&(mapped[:,0]<paper_w+12)&
                      (mapped[:,1]>-12)&(mapped[:,1]<paper_h+12))
            if int(inside.sum()) < 1:
                return None, 'close mode: page outside current view'
            return TrackResult(True, corners, H_live_mm, np.linalg.inv(H_live_mm),
                               'ORB-CLOSE', f'{count}/{len(matches)} median={err:.1f}px',
                               len(matches), count, err), ''
        except (ValueError, OverflowError, np.linalg.LinAlgError, cv2.error):
            return None, 'close mode: invalid homography'

    def _stable(self, candidate, frame):
        # Unlike full-page corners, which may be FAR outside close-range FOV,
        # compare correspondence of 5 *visible* frame anchors in paper mm.
        h, w = frame.shape[:2]
        anchors = np.float32([[w/2,h/2],[w*.25,h*.3],[w*.75,h*.3],
                              [w*.25,h*.7],[w*.75,h*.7]])
        if self.last_good is not None:
            try:
                before = _warp(anchors, self.last_good.H_live_to_mm)
                after = _warp(anchors, candidate.H_live_to_mm)
                movement = float(np.median(np.linalg.norm(before-after,axis=1)))
            except (cv2.error, ValueError):
                movement = float('inf')
            if not math.isfinite(movement) or movement > 40.0:
                self.last_good = None
                self.pending = None
                self.pending_count = 0
                self.last_reason = f'camera moved quickly ({movement:.1f}mm); reacquiring'
                return None
        if self.pending is not None:
            try:
                before = _warp(anchors, self.pending.H_live_to_mm)
                after = _warp(anchors, candidate.H_live_to_mm)
                motion = float(np.median(np.linalg.norm(before-after,axis=1)))
            except (cv2.error, ValueError):
                motion = float('inf')
            self.pending_count = self.pending_count+1 if motion < 18 else 1
        else:
            self.pending_count = 1
        self.pending = candidate
        if self.last_good is None and self.pending_count < 2:
            self.last_reason = f'stabilizing {self.pending_count}/2'
            return None
        self.last_good = candidate
        self.fail_frames = 0
        self.orb_ever = True
        self.last_reason = ''
        return candidate
