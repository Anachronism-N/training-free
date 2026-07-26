#!/usr/bin/env python3
"""Parse temporal jump log into partial CSV for analysis."""
import re, csv, sys
from collections import Counter

log_path = sys.argv[1]
csv_path = sys.argv[2]

rows = []
with open(log_path) as f:
    for line in f:
        m = re.search(r'score=([0-9.]+)\s+video=(\S+)', line)
        if m:
            rows.append({'video': m.group(2), 'temporal_jump': m.group(1)})

print(f'Parsed {len(rows)} entries from log')
if rows:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video', 'temporal_jump'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {csv_path}')
    methods = Counter()
    for r in rows:
        parts = r['video'].split('/')
        for p in parts:
            if 'merge' in p or 'recent' in p or 'native' in p:
                methods[p] += 1
                break
    for m, c in sorted(methods.items()):
        print(f'  {m}: {c} videos')
else:
    print('No entries found!')
