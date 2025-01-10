.venv/scripts/activate.ps1
cd src
&python -m ellishape_cli -i data/circle.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/square.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/square_rotate.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/star.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/leaf.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/leaf2.png -n 35 -n_dots 500
&python -m ellishape_cli -i data/leaf3.png -n 35 -n_dots 500
cd ..