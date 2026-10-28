#!/usr/bin/env python3
"""Compare corresponding decoded RGB videos; every case must exceed 25 dB."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

THRESHOLD_DB = 25.0
CASES = ('unitree_z1_stackbox', 'unitree_z1_dual_arm_stackbox',
         'unitree_z1_dual_arm_stackbox_v2', 'unitree_z1_dual_arm_cleanup_pencils',
         'unitree_g1_pack_camera')
DEFAULT_REFERENCE = '/share/results/unifolm/fp32/reference/outputs'


def psnr(squared_error, pixels):
    if pixels <= 0 or not math.isfinite(squared_error) or squared_error < 0:
        raise ValueError('Invalid decoded pixel statistics')
    # JSON has no infinity: null + exact_match=true encodes infinite PSNR.
    return None if squared_error == 0 else 10 * math.log10(255.0**2 / (squared_error / pixels))


def video_files(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Missing video directory or symlink: ' + str(root))
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Video tree contains a symlink: ' + str(path))
        if path.is_file() and path.suffix.lower() == '.mp4':
            files[path.relative_to(root).as_posix()] = path
    if not files:
        raise ValueError('No MP4 videos in ' + str(root))
    return files


def compare_video(actual, reference):
    import av
    import numpy as np
    squared_error = 0.0
    pixels = frames = 0
    with av.open(str(actual)) as a, av.open(str(reference)) as r:
        if len(a.streams.video) != 1 or len(r.streams.video) != 1:
            raise ValueError('Expected exactly one video stream')
        for af, rf in itertools.zip_longest(a.decode(video=0), r.decode(video=0)):
            if af is None or rf is None:
                raise ValueError('Frame count mismatch: ' + str(actual))
            x, y = af.to_ndarray(format='rgb24'), rf.to_ndarray(format='rgb24')
            if x.shape != y.shape:
                raise ValueError('Frame shape mismatch: ' + str(actual))
            diff = x.astype(np.float64) - y.astype(np.float64)
            squared_error += float(np.sum(diff * diff, dtype=np.float64))
            pixels += int(x.size)
            frames += 1
    value = psnr(squared_error, pixels)
    return {'frames':frames, 'rgb_values':pixels, 'squared_error':squared_error,
            'psnr_db':value, 'exact_match':squared_error == 0}


def validate_videos(result_dir, reference_dir, expected_cases=CASES):
    # Validate every case independently: one missing/corrupt case never erases others.
    per_video, cases = {}, {}
    for case in expected_cases:
        item = {'psnr_db':None, 'exact_match':False, 'passed':False,
                'video_count':0, 'frames':0, 'reason':None}
        try:
            if Path(result_dir).is_symlink() or Path(reference_dir).is_symlink():
                raise ValueError('Video root must not be a symlink')
            actual = video_files(Path(result_dir)/case)
            reference = video_files(Path(reference_dir)/case)
            if set(actual) != set(reference):
                raise ValueError('Video set mismatch; missing=%s extra=%s' %
                                 (sorted(set(reference)-set(actual)), sorted(set(actual)-set(reference))))
            stats = []
            for name in sorted(reference):
                stat = compare_video(actual[name], reference[name])
                per_video[case+'/'+name] = stat
                stats.append(stat)
            error = sum(stat['squared_error'] for stat in stats)
            pixels = sum(stat['rgb_values'] for stat in stats)
            value = psnr(error, pixels)
            item.update(psnr_db=value, exact_match=error==0,
                        passed=value is None or value>THRESHOLD_DB,
                        video_count=len(stats), frames=sum(stat['frames'] for stat in stats),
                        mse=error/pixels)
            if not item['passed']:item['reason']='PSNR must be strictly greater than 25 dB'
        except Exception as error:
            item['reason']=str(error)
        cases[case]=item
    return {'metric':'psnr_rgb_uint8', 'threshold_db':THRESHOLD_DB,
            'comparison':'strictly_greater', 'aggregation':'case_pooled_rgb_mse',
            'reference_dir':str(reference_dir),
            'video_count':sum(item['video_count'] for item in cases.values()),
            'psnr_ok':bool(cases) and all(item['passed'] for item in cases.values()),
            'cases':cases, 'per_video':per_video}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--reference', default=DEFAULT_REFERENCE, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        report = validate_videos(args.results, args.reference)
    except Exception as error:
        report = {'metric':'psnr_rgb_uint8', 'threshold_db':THRESHOLD_DB,
                  'psnr_ok':False, 'reason':str(error), 'cases':{}}
    report['validation_wall_s'] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, value in report['cases'].items():
        display = 'inf' if value['exact_match'] else ('nan' if value['psnr_db'] is None else format(value['psnr_db'], '.17g'))
        print('PSNR case=%s psnr_db=%s threshold_db=25 pass=%s' %
              (name, display, str(value['passed']).lower()), flush=True)
        if value.get('reason'):print('PSNR_CASE_ERROR case=%s reason=%s' % (name,value['reason']),flush=True)
    if not report['psnr_ok']:
        print('PSNR_FAILED: ' + report.get('reason', 'Every case must be strictly greater than 25 dB'), flush=True)
    return 0 if report['psnr_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
