from pathlib import Path

p = Path('docs/spec_trade_ingestion_v1.2.md')
lines = p.read_text(encoding='utf-8').splitlines()
hits = []
for i, l in enumerate(lines, 1):
    if '0=Valid' in l or 'Code 0' in l or 'Code 1' in l or 'categorical' in l or 'observed' in l:
        hits.append((i, l))
out = '\n'.join([f'{i}: {l}' for i, l in hits])
Path('docs/reports/_spec_semantics_hits.txt').write_text(out, encoding='utf-8')
print('WROTE docs/reports/_spec_semantics_hits.txt')
