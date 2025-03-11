import numpy as np
from skbio import DistanceMatrix, nj
from sys import argv

a = np.loadtxt(argv[1], delimiter=',', dtype=str)
b = a[1:,1:].astype(np.float64)
b += b.T
np.fill_diagonal(b, 0)
a[1:, 1:] = b
names = a[1:,0]

dis_matrix = DistanceMatrix(b, names)
tree = nj(dis_matrix)
tree.write('tree.nwk', 'newick')
np.savetxt('new.csv', a, delimiter=',', fmt='%s')
