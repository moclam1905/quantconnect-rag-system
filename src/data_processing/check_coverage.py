import json, sys, glob, re, pathlib

structured = pathlib.Path(sys.argv[1])              # 1st arg
sections   = pathlib.Path(sys.argv[2])              # 2nd arg = folder

with open(structured, 'r', encoding='utf-8') as f:
    blocks = [b for b in json.load(f)['content_blocks']
              if not b['content_type'].startswith('heading_')]
src_orders = {b['order'] for b in blocks}

dst_orders = set()
for path in glob.glob(str(sections / '*.json')):
    with open(path, 'r', encoding='utf-8') as f:
        sec = json.load(f)
    for c in sec['chunks']:
        m = re.match(r'chunk_(\d+)', c['chunk_id'])
        if m:
            dst_orders.add(int(m.group(1)))

missing = src_orders - dst_orders
extra   = dst_orders - src_orders

print(f"blocks in source : {len(src_orders)}")
print(f"chunks in output : {len(dst_orders)}")
print(f"missing          : {len(missing)}")
print(f"extra            : {len(extra)}")
if missing:
    print("first few missing orders:", sorted(missing)[:10])
