"""v7.5: safer close-to-paper tracking fallback for a handheld phone.

Default tracker is exactly the tested v7.3 close-range tracker. When that
fails to match a *partly visible* known page, retry ORB with a larger reference
image to reduce scale mismatch against the close-up camera image.

No new geometry is invented; both branches require RANSAC, spatially dispersed
inliers, two stable frames and a visible target before navigation.
Neither branch guarantees tracking from very small or blurry crops.
"""
from __future__ import annotations
from document_tracker_v7_3 import CloseRangePaperTracker


class ElbowClosePaperTracker(CloseRangePaperTracker):
    """Dual-reference-resolution known-document tracker, primary unchanged."""

    def __init__(self, manifest, reference_path, max_dim=1080, detail_max_dim=1600):
        super().__init__(manifest, reference_path, max_dim=max_dim)
        self.detail_max_dim = int(detail_max_dim)
        self.detail = None
        if self.detail_max_dim > max_dim:
            self.detail = CloseRangePaperTracker(
                manifest, reference_path, max_dim=self.detail_max_dim
            )

    def orb_match(self, frame):
        # The previously verified path always gets first chance.
        candidate, reason = super().orb_match(frame)
        if candidate is not None or self.detail is None:
            return candidate, reason
        # Preserve the acquired pose for the close-mode geometry gate.
        # The detail object's update() is NOT invoked, so this fallback still
        # goes through this object's _stable() and never bypasses stability.
        self.detail.last_good = self.last_good
        candidate, fine_reason = self.detail.orb_match(frame)
        if candidate is not None:
            candidate.method = 'ORB-CLOSE-HR' if candidate.method != 'ORB' else 'ORB-HR'
            return candidate, ''
        return None, f'{reason} | detailed-reference: {fine_reason}'
