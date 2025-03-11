from numba import njit, float64, jit
from scipy.spatial.distance import squareform, pdist
from timeit import default_timer as timer
import numpy
import torch

a = timer()
n = 10000
n_dot = 1024
mat = numpy.random.randint(0, 10, size=n*n_dot*2)
mat = mat.reshape((n, n_dot*2)).astype(numpy.float64)


def s_func(mat):
    s_pdist = pdist(mat)
    return s_pdist


@njit(cache=True)
def s_func2(X: numpy.ndarray) -> numpy.ndarray:
    m = X.shape[0]
    n = X.shape[1]
    result = numpy.empty((m * (m - 1)) // 2, dtype=X.dtype)  # Pre-allocate
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            distance = 0.0
            for l in range(n):
                diff = X[i, l] - X[j, l]
                distance += diff * diff
            result[k] = numpy.sqrt(distance)
            k += 1
    return result


b = timer()
s_pdist = s_func(mat)

c = timer()
mat2 = torch.from_numpy(mat).cuda()
cc = timer()
t_pdist = torch.nn.functional.pdist(mat2)
d = timer()
s_result = squareform(s_pdist)
t_result = squareform(t_pdist.cpu())
result = numpy.strings.mod('%.4f', s_result)
#print(s_result)
diff = numpy.sum(s_result-t_result)
e = timer()


print('init', b-a)
print('scipy', c-b)
print('move', cc-c)
print('torch', d-cc)
print('square',e-d)
print('result')
print(result)
print('average', numpy.average(s_result))
print('difference', diff)

b2 = timer()
s_func2(mat)
c2 = timer()
print('scipy numba', c2-b2)