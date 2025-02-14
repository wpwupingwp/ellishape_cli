.venv/scripts/activate.ps1
# &python src/ellishape_cli/reconstruct.py src/data/ref/ref_efd.csv
&python src/ellishape_cli/cli.py -i "src/data/Quercus imbricaria.png" -n 35 -n_dots 512 -out_image -out verify.1.csv
&python src/ellishape_cli/cli.py -i "src/data/Quercus alba.png" -n_harmonic 35 -n_dots 512 -out_image -out verify.2.csv
