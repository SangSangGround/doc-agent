"""Summarize operator-triggered touch-to-target discrepancies. NOT physical localization accuracy.

Usage from project root:
  .\\.venv_scan\\Scripts\\python.exe detection\\tracking_eval_report.py --csv detection\\refinement_output\\document2\\writable_output\\tracking_touch_samples.csv
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile_linear(values, p):
    vals = sorted(values)
    if not vals:
        raise ValueError('Empty samples')
    pos = (len(vals) - 1) * p
    lower, upper = math.floor(pos), math.ceil(pos)
    return vals[lower] * (upper - pos) + vals[upper] * (pos - lower) if upper != lower else vals[lower]


def summarize(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    per_field = defaultdict(list)
    for i, r in enumerate(rows, 1):
        try:
            distance = float(r['distance_mm_display'])
            if not math.isfinite(distance) or distance < 0:
                raise ValueError('nonnegative finite distance required')
            per_field[(r['field_id'], r.get('label', ''))].append(distance)
        except (KeyError, ValueError, TypeError) as exc:
            print(f'Skipping malformed data row {i}: {exc}')
    if not per_field:
        print('No valid measurement rows. In the camera window press M while physically touching a target.')
        return {}
    for (fid, label), vals in sorted(per_field.items()):
        print(f'{fid} {label}: n={len(vals)}, median={statistics.median(vals):.1f} mm, '
              f'p90={percentile_linear(vals,.90):.1f} mm, max={max(vals):.1f} mm '
              f'{"(fewer than 5 samples)" if len(vals)<5 else ""}')
    pooled = [v for vals in per_field.values() for v in vals]
    print(f'All operator-declared touches: n={len(pooled)}, median={statistics.median(pooled):.1f} mm, '
          f'p90={percentile_linear(pooled,.9):.1f} mm')
    print('INTERPRETATION: finger-landmark to approved-coordinate discrepancy when the operator says the finger touches the spot. '
          'Includes pointing placement, landmark, parallax and calibration errors. NOT independently verified physical accuracy.')
    return dict(per_field)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    a = p.parse_args()
    if not a.csv.is_file():
        p.error(f'Measurement CSV not found: {a.csv}')
    summarize(a.csv)
