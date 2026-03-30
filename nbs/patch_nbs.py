import json, glob

for path in glob.glob('nbs/**/*.ipynb', recursive=True) + glob.glob('nbs/*.ipynb'):
    with open(path) as f:
        nb = json.load(f)
    nb.setdefault('metadata', {})['skip_exec'] = True
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Patched {path}")