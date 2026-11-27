"""Automatic paper tracking for reviewed A4 navigation targets (developer PoC).

Preferred method: ORB + BFMatcher + RANSAC homography, reference original
photo -> each live frame. Restricts reference features to paper and rejects
low-coverage, low-inlier, implausible or unstable homographies.

Optional --allow-contour enables a weaker page-contour fallback. The contour
fallback has NO document identity or orientation guarantee; use only with the
same single, upright paper, an uncluttered dark desk, and visually inspect
crosshairs. Uncertain results always pause guidance.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np


def read_exif_bgr(path: Path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resize_limit(im, max_dim):
    h, w = im.shape[:2]
    sf = min(1., float(max_dim) / max(w, h))
    nw, nh = max(1, round(w*sf)), max(1, round(h*sf))
    out = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA) if (nw,nh)!=(w,h) else im
    to_small = np.diag([nw/w, nh/h, 1.]).astype(np.float64)
    return out, to_small


def _preprocess(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)


def _warp(pts, H):
    return cv2.perspectiveTransform(np.asarray(pts,np.float64).reshape(-1,1,2),
                                    np.asarray(H,np.float64)).reshape(-1,2)


def _valid_quad(quad, width, height, min_area_fraction=.07):
    p = np.asarray(quad,np.float32)
    if p.shape!=(4,2) or not np.isfinite(p).all():
        return False
    if np.min(p[:,0]) < -0.03*width or np.max(p[:,0]) > 1.03*width:
        return False
    if np.min(p[:,1]) < -0.03*height or np.max(p[:,1]) > 1.03*height:
        return False
    if not cv2.isContourConvex(p) or cv2.contourArea(p) < min_area_fraction*width*height:
        return False
    edges=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    if np.min(edges) < 0.09*min(width,height) or max(edges)/min(edges) > 6.:
        return False
    return True


def _ordered_rect(points):
    p=np.asarray(points,np.float32).reshape(4,2)
    s=p[:,0]+p[:,1]; d=p[:,1]-p[:,0]
    indices=[int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    if len(set(indices))!=4:
        return None
    return p[indices]


def find_paper_contour(frame):
    """Best-effort border quad; NOT a verification of document identity."""
    h,w=frame.shape[:2]
    small, T=resize_limit(frame, 950)
    sh,sw=small.shape[:2]
    gray=cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    edge=cv2.Canny(gray,40,120)
    edge=cv2.morphologyEx(edge, cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=2)
    contours,_=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area < 0.14*sw*sh:
            continue
        perimeter=cv2.arcLength(c,True)
        poly=cv2.approxPolyDP(c,0.025*perimeter,True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):
            continue
        q=_ordered_rect(poly.reshape(4,2))
        if q is None or not _valid_quad(q,sw,sh,.13):
            continue
        top=np.linalg.norm(q[1]-q[0]); bottom=np.linalg.norm(q[2]-q[3]);
        left=np.linalg.norm(q[3]-q[0]); right=np.linalg.norm(q[2]-q[1])
        r=(left+right)/max(1.,top+bottom)
        # Upright portrait only: suppress landscape / near-90-degree rotation.
        if r < 1.07 or r > 3.8:
            continue
        candidates.append((area,q))
    if not candidates:
        return None
    q=max(candidates,key=lambda x:x[0])[1]
    result=_warp(q,np.linalg.inv(T))
    return result if _valid_quad(result,w,h,.12) else None


@dataclass
class TrackResult:
    valid: bool
    corners: np.ndarray | None = None
    H_live_to_mm: np.ndarray | None = None
    H_mm_to_live: np.ndarray | None = None
    method: str = 'NONE'
    detail: str = ''
    matched: int = 0
    inliers: int = 0
    reproject_error_px: float = float('inf')


class PaperTracker:
    def __init__(self, manifest, reference_path, allow_contour=False, max_dim=1080,
                 min_inliers=16, min_inlier_ratio=.36, max_jump_fraction=.17):
        self.manifest=manifest
        self.ref_path=Path(reference_path).expanduser().resolve()
        if not self.ref_path.is_file():
            raise FileNotFoundError(f'Reference original photo not found: {self.ref_path}')
        required_sha=manifest.get('source_image_sha256')
        if required_sha and sha256_file(self.ref_path)!=required_sha:
            raise ValueError('Reference photo differs from navigation manifest (SHA256 mismatch). Use ORIGINAL document2.jpg.')
        self.ref=read_exif_bgr(self.ref_path)
        self.ref_h,self.ref_w=self.ref.shape[:2]
        self.ref_corners=np.asarray(manifest['reference_corners_px_TL_TR_BR_BL'],np.float64)
        if not _valid_quad(self.ref_corners,self.ref_w,self.ref_h,.06):
            raise ValueError('Manifest reference corners are invalid')
        self.H_ref_to_mm=np.asarray(manifest['homography_reference_px_to_mm'],np.float64)
        if self.H_ref_to_mm.shape!=(3,3) or not np.isfinite(self.H_ref_to_mm).all():
            raise ValueError('Manifest reference-to-mm matrix is invalid')
        self.ref_small,self.ref_to_small=resize_limit(self.ref,max_dim)
        gray=_preprocess(cv2.cvtColor(self.ref_small,cv2.COLOR_BGR2GRAY))
        mask=np.zeros(gray.shape,np.uint8)
        q=_warp(self.ref_corners,self.ref_to_small)
        cv2.fillConvexPoly(mask,np.round(q).astype(np.int32),255)
        mask=cv2.erode(mask,np.ones((11,11),np.uint8),iterations=1)
        self.orb=cv2.ORB_create(nfeatures=3500,scaleFactor=1.2,nlevels=8,edgeThreshold=20,fastThreshold=11)
        self.kp_ref,self.desc_ref=self.orb.detectAndCompute(gray,mask)
        if self.desc_ref is None or len(self.kp_ref)<35:
            raise ValueError('Too few visual features on reference page for automatic tracking')
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
        self.allow_contour=allow_contour
        self.max_dim=max_dim
        self.min_inliers=min_inliers
        self.min_ratio=min_inlier_ratio
        self.max_jump=max_jump_fraction
        self.last_good=None
        self.pending=None
        self.pending_count=0
        self.fail_frames=0
        self.orb_ever=False
        self.last_reason='waiting for paper'

    def orb_match(self,frame):
        h,w=frame.shape[:2]
        small, frame_to_small=resize_limit(frame,self.max_dim)
        gray=_preprocess(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY))
        kps,desc=self.orb.detectAndCompute(gray,None)
        if desc is None or len(kps)<25:
            return None,'too few live features'
        pairs=self.matcher.knnMatch(self.desc_ref,desc,k=2)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance < .76*b.distance]
        if len(good)<self.min_inliers:
            return None,f'ORB good matches: {len(good)}'
        pts_ref=np.float32([self.kp_ref[m.queryIdx].pt for m in good])
        pts_live=np.float32([kps[m.trainIdx].pt for m in good])
        Hsmall,inlier_mask=cv2.findHomography(pts_ref,pts_live,cv2.RANSAC,5.0,maxIters=2500,confidence=.995)
        if Hsmall is None or inlier_mask is None:
            return None,'RANSAC no matrix'
        inl=inlier_mask.ravel().astype(bool)
        nin=int(inl.sum())
        ratio=nin/len(good)
        if nin<self.min_inliers or ratio<self.min_ratio:
            return None,f'ORB weak geometry: inliers {nin}/{len(good)}'
        # Require spatial coverage of the *document*, not one word / stamp.
        p_ref=pts_ref[inl]
        ref_quad_small=_warp(self.ref_corners,self.ref_to_small)
        xs=ref_quad_small[:,0];ys=ref_quad_small[:,1]
        coverage_x=np.ptp(p_ref[:,0])/max(1.,np.ptp(xs))
        coverage_y=np.ptp(p_ref[:,1])/max(1.,np.ptp(ys))
        if coverage_x<.27 or coverage_y<.27:
            return None,f'ORB matches clustered ({coverage_x:.2f}, {coverage_y:.2f})'
        proj=_warp(pts_ref[inl],Hsmall)
        err=np.linalg.norm(proj-pts_live[inl],axis=1)
        median_err=float(np.median(err))
        if median_err>3.7:
            return None,f'RANSAC median error {median_err:.1f}px'
        try:
            H_ref_to_live=np.linalg.inv(frame_to_small)@Hsmall@self.ref_to_small
            corners=_warp(self.ref_corners,H_ref_to_live)
            if not _valid_quad(corners,w,h,.065):
                return None,'projected page quadrilateral invalid/out of view'
            H_live_to_mm=self.H_ref_to_mm@np.linalg.inv(H_ref_to_live)
            if not np.isfinite(H_live_to_mm).all():
                return None,'nonfinite mapping'
            if abs(np.linalg.det(H_live_to_mm)) < 1e-14:
                return None,'near singular transform'
        except (np.linalg.LinAlgError,ValueError):
            return None,'singular transformation'
        result=TrackResult(True,corners,H_live_to_mm,np.linalg.inv(H_live_to_mm),
                           'ORB',f'ORB {nin}/{len(good)} median={median_err:.1f}px',len(good),nin,median_err)
        return result,''

    def contour_match(self,frame):
        q=find_paper_contour(frame)
        if q is None:
            return None,'page contour not found'
        h,w=frame.shape[:2]
        # If ORB ever established the identity, don't switch to an unrelated rectangle
        # far from the last matched paper.
        if self.orb_ever and self.last_good is not None:
            d=np.mean(np.linalg.norm(q-self.last_good.corners,axis=1))/math.hypot(w,h)
            if d>.10:
                return None,'page contour differs from verified document position'
        mm=self.manifest
        dest=np.float32([[0,0],[mm['paper_width_mm'],0],
                         [mm['paper_width_mm'],mm['paper_height_mm']],[0,mm['paper_height_mm']]])
        H=cv2.getPerspectiveTransform(np.float32(q),dest)
        return TrackResult(True,q,H,np.linalg.inv(H),'CONTOUR',
                           'Contour only: paper identity/upright orientation assumed'),''

    def _stable(self,candidate,frame):
        h,w=frame.shape[:2]
        previous=self.last_good
        if previous is not None:
            disp=np.mean(np.linalg.norm(candidate.corners-previous.corners,axis=1))/math.hypot(w,h)
            # Camera moved abruptly / possible wrong match: require reacquisition.
            if disp>self.max_jump:
                self.last_good=None
                self.pending=None
                self.pending_count=0
                self.last_reason='large pose jump - rechecking'
                return None
        if self.pending is None:
            self.pending=candidate
            self.pending_count=1
        else:
            dist=np.mean(np.linalg.norm(candidate.corners-self.pending.corners,axis=1))/math.hypot(w,h)
            if dist<.08 and self.pending.method==candidate.method:
                self.pending_count+=1
            else:
                self.pending_count=1
            self.pending=candidate
        # Require first lock across 2 frames (contour-only needs 4) to reject spurious fit.
        needed=4 if candidate.method=='CONTOUR' and not self.orb_ever else 2
        if self.last_good is None and self.pending_count<needed:
            self.last_reason=f'stabilizing {self.pending_count}/{needed}'
            return None
        self.last_good=candidate
        self.fail_frames=0
        self.last_reason=''
        if candidate.method=='ORB':
            self.orb_ever=True
        return candidate

    def update(self,frame):
        """Return None when geometry isn't currently safe; never reuse stale H."""
        candidate,reason=self.orb_match(frame)
        if candidate is None and self.allow_contour:
            candidate,reason2=self.contour_match(frame)
            if candidate is None:
                reason += '; '+reason2
        if candidate is None:
            self.last_reason=reason
            self.fail_frames+=1
            self.pending=None
            self.pending_count=0
            if self.fail_frames>3:
                self.last_good=None
            return None
        found=self._stable(candidate,frame)
        return found
