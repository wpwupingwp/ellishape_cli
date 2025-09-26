from pathlib import Path
from sys import argv

raw = Path(argv[1]).resolve()
out_dir = raw.parent / 'split'
if out_dir.exists():
    raise ValueError(f'{out_dir} exists')
out_dir.mkdir()
name_lines = dict()
with open(raw, 'r') as _:
    # skip head
    head = _.readline()
    for line in _:
        fields = line.split('/')[-1]
        filename = fields.split(',')[0]
        species = filename.split(' ')[:2]
        species_name = '_'.join(species)
        if species_name in name_lines:
            name_lines[species_name].append(line)
        else:
            name_lines[species_name] = [line, ]
for name in name_lines:
    out_file = out_dir / (name+'.dot.csv')
    with open(out_file, 'w') as out:
        out.write(head)
        out.write(''.join(name_lines[name]))
