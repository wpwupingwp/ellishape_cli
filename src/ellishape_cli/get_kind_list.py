from sys import argv
import numpy as np
from pathlib import Path

input_file = Path(argv[1])
a_raw = np.loadtxt(input_file, dtype=str, delimiter=',',quotechar='"')
# a = np.loadtxt(input_file, dtype=str, delimiter=',',quotechar='"', skiprows=1)
a = a_raw[1:, :]
# np.random.shuffle(a)
idx = np.arange(a.shape[0])+1

# D:\\dataset\\hs_2024419\\Castanea sativa\\Castanea sativa Mill._730884840_13.png
# change code according to format
a1 = np.strings.replace(a[:, 0], '\\', '_')
a2 = np.strings.replace(a1, 'D:_dataset_hs_2024419_', '')
# a3 = np.strings.partition(a2, '_')[0]
a3 = np.strings.partition(a2, '(')[0]

out = np.stack([idx, a3]).T
a[:, 0] = idx
a = np.vstack([a_raw[0], a])
out = np.vstack([['name', 'kind'], out])
np.savetxt(input_file.with_suffix('.simple.csv'), a, delimiter=',', fmt='%s')
np.savetxt(input_file.with_suffix('.kind.csv'), out, delimiter=',', fmt='%s')
