.venv/scripts/activate.ps1
&python src/ellishape_cli/reconstruct.py src/data/ref/ref_efd.csv
&python src/ellishape_cli/cli.py -i "src/data/ref/S5_Quercus imbricaria_1.png" -n 35 -n_dots 400 -out_image -out verify.1
&python src/ellishape_cli/cli.py -i "src/data/ref/S10_Quercus alba_1.png" -n_harmonic 35 -n_dots 400 -out_image -out verify.2
