from pathlib import Path
import csv
p=Path('data/processed/meta/node_index.csv')
if not p.exists():
    print('MISSING')
    raise SystemExit(2)
rows=list(csv.DictReader(p.open('r',encoding='utf-8')))
print('rows',len(rows))
ids=[int(r['node_id']) for r in rows]
print('min',min(ids),'max',max(ids))
print('unique_iso3',len({r['iso3'].strip().upper() for r in rows}))
if len(rows)!=194:
    print('BAD_COUNT')
    raise SystemExit(2)
if set(ids)!=set(range(194)):
    print('BAD_IDS')
    raise SystemExit(2)
print('OK')

