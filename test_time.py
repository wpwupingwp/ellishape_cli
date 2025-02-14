from timeit import default_timer as timer
from subprocess import run
from pathlib import Path
files=("circle.png","square.png","squarre_rotate.png","leaf.png","leaf2.png")
files=("leaf.png", )
args = [
    '-n_dots 512 -n 35 -method dots',
    '-n_dots 512 -n 35 -method chain_code',
    '-n_dots 512 -n 35 -method dots -out_image',
    '-n_dots 512 -n 35 -method chain_code -out_image',
    '-n_dots 512 -n 64 -method dots -out ../data/512_64',
    '-n_dots 512 -n 64 -method dots -out_image -out ../data/512_64',
    '-n_dots 4096 -n 64 -method dots -out ../data/4096_64',
    '-n_dots 4096 -n 64 -method dots -out_image -out ../data/4096_64'
]
log = open('log.txt', 'w')
for f in files:
    f = Path(f'../data/{f}').resolve()
    for a in args:
        start = timer()
        cmd = f'python cli.py -i {f} '
        command = cmd + a
        run(command, shell=True, stdout=log, stderr=log)
        end = timer()
        print(f'{f.name},{a},{end-start:.4f}')