"""Build the result tables (Markdown + CSV) from <output>/per_pair.jsonl.

    python -m urban3d_eval.report [--output urban3d_eval/results]

Scores are means over image pairs (macro average, like `summarize` in
ORB/metrics/local_feature_metrics.py), fractions in [0, 1].
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from urban3d_eval.evaluate import DEFAULT_OUTPUT

TILT_BINS = ((0, 7, 'tilt 0–7°'), (7, 14, 'tilt 7–14°'), (14, 181, 'tilt ≥ 14°'))   # by absolute value


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows, thresholds):
    """One dict of metrics for a group of rows (pairs) of a single method."""
    out = {'pairs': len(rows)}
    for t_index, t in enumerate(thresholds):
        out[f'MMA@{t:g}'] = float(np.mean([r['mma'][t_index] for r in rows]))
    for t_index, t in enumerate(thresholds):
        out[f'MHA@{t:g}'] = float(np.mean([r['mha'][t_index] for r in rows]))
    out['mean_keypoints'] = float(np.mean([(r['keypoints_a'] + r['keypoints_b']) / 2 for r in rows]))
    out['mean_matches'] = float(np.mean([r['matches'] for r in rows]))
    out['scored_matches_ratio'] = float(sum(r['scored_matches'] for r in rows)
                                        / max(sum(r['matches'] for r in rows), 1))
    out['extract_ms_per_image'] = float(np.mean([(r['extract_ms_a'] + r['extract_ms_b']) / 2 for r in rows]))
    return out


def group_rows(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row['method'], {}).setdefault(key(row), []).append(row)
    return groups


def tilt_bin(row):
    angle = abs(row['tilt_deg'])
    return next(label for lo, hi, label in TILT_BINS if lo <= angle < hi)


def _md_table(header, body):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in body]
    return '\n'.join(lines)


def _fmt(value, kind):
    if kind == 'ratio':
        return f'{value:.3f}'
    if kind == 'int':
        return f'{value:.0f}'
    return f'{value:.1f}'


def build_report(rows, thresholds):
    methods = list(dict.fromkeys(r['method'] for r in rows))
    overall = {m: summarize([r for r in rows if r['method'] == m], thresholds) for m in methods}
    by_city = group_rows(rows, lambda r: r['city'])
    by_tilt = group_rows(rows, tilt_bin)
    pair_counts = {m: overall[m]['pairs'] for m in methods}
    tiles = len({r['tile'] for r in rows})
    mma_cols = [f'MMA@{t:g}' for t in thresholds]
    mha_cols = [f'MHA@{t:g}' for t in thresholds]

    md = ['# Urban3D: MMA / MHA', '',
          f'{tiles} tiles, {max(pair_counts.values())} pairs (source.png ↔ aug_00…05) per method. '
          'Scores are fractions in [0, 1], mean over pairs; thresholds in pixels.', '']
    if len(set(pair_counts.values())) > 1:
        md += [f'> **Warning:** methods have different numbers of pairs: {pair_counts}', '']

    md += ['## Overall', '',
           _md_table(['method'] + mma_cols + mha_cols,
                     [[m] + [_fmt(overall[m][c], 'ratio') for c in mma_cols + mha_cols] for m in methods]), '',
           '## Cost', '',
           _md_table(['method', 'keypoints / image', 'matches / pair', 'matches with GT', 'ms / image'],
                     [[m, _fmt(overall[m]['mean_keypoints'], 'int'), _fmt(overall[m]['mean_matches'], 'int'),
                       _fmt(overall[m]['scored_matches_ratio'], 'ratio'),
                       _fmt(overall[m]['extract_ms_per_image'], 'ms')] for m in methods]), '',
           '`ms / image` is the wall time of the extractor inside the evaluation process; '
           'it is inflated when several evaluations share the CPU, so compare methods only '
           'within one run.', '']

    mid = [c for c in (f'{t:g}' for t in thresholds) if c in ('1', '3', '5')] or [f'{t:g}' for t in thresholds]
    for title, groups, labels in (
            ('By city', by_city, sorted({r['city'] for r in rows})),
            ('By vertical tilt of the target (absolute value)', by_tilt, [label for _, _, label in TILT_BINS])):
        header = ['method', 'group', 'pairs'] + [f'MMA@{t}' for t in mid] + [f'MHA@{t}' for t in mid]
        body = []
        for m in methods:
            for label in labels:
                group = groups.get(m, {}).get(label)
                if not group:
                    continue
                s = summarize(group, thresholds)
                body.append([m, label, str(s['pairs'])] +
                            [_fmt(s[f'MMA@{t}'], 'ratio') for t in mid] +
                            [_fmt(s[f'MHA@{t}'], 'ratio') for t in mid])
        md += [f'## {title}', '', _md_table(header, body), '']

    summary_rows = []
    for m in methods:
        summary_rows.append({'method': m, 'group': 'all', **overall[m]})
        for scope, groups in (('city', by_city), ('tilt', by_tilt)):
            for label, group in groups[m].items():
                summary_rows.append({'method': m, 'group': f'{scope}:{label}', **summarize(group, thresholds)})
    return '\n'.join(md), summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_path = args.output / 'per_pair.jsonl'
    if not rows_path.is_file():
        parser.error(f'{rows_path} not found; run urban3d_eval.evaluate first.')
    rows = read_rows(rows_path)
    if not rows:
        parser.error(f'{rows_path} is empty.')
    thresholds = json.loads((args.output / 'run_config.json').read_text())['thresholds_px']

    markdown, summary_rows = build_report(rows, thresholds)
    (args.output / 'results.md').write_text(markdown + '\n')

    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    fields = ['method', 'tile', 'city', 'variant', 'roll_deg', 'tilt_deg', 'keypoints_a', 'keypoints_b',
              'matches', 'scored_matches', 'median_error_px', 'reference_inlier_ratio',
              'extract_ms_a', 'extract_ms_b']
    fields += [f'MMA@{t:g}' for t in thresholds] + [f'MHA@{t:g}' for t in thresholds]
    with (args.output / 'per_pair.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields if k in row}
            flat.update({f'MMA@{t:g}': v for t, v in zip(thresholds, row['mma'])})
            flat.update({f'MHA@{t:g}': v for t, v in zip(thresholds, row['mha'])})
            writer.writerow(flat)
    print(markdown)


if __name__ == '__main__':
    main()
