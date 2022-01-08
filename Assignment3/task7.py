import numpy as np


d = 4
k = 2

def task_a(eigen_values):
    sorted = np.sort(eigen_values)
    return np.sum(sorted[:d-k])


def task_b(eigen_values, eigen_vectors):
    arg_sort = np.argsort(eigen_values)
    U = np.flip(eigen_vectors[: ,arg_sort[d-k:]], axis=1)

    print(f'The first vector norm:{np.linalg.norm(U[:, 0])}')
    print(f'The second vector norm:{np.linalg.norm(U[:, 1])}')
    print(f'The inner product of the 2 vectors:{U[:, 0] @ U[:, 1]}')
    return U

def task_c(U, X):
    restored_X =np.asarray([U @ U.T @ X[i] for i in range(3)])
    distortion = np.linalg.norm(X - restored_X) ** 2
    return restored_X, distortion


X = np.asarray([[1, -2, 5, 4], [3, 2, 1, -5], [-10, 1, -4, 6]])
A = X.T @ X
eigen_values, eigen_vectors = np.linalg.eig(A)

distortion = task_a(eigen_values)
print(f'The distortion is: {distortion}')
U = task_b(eigen_values, eigen_vectors)
print(f'The matrix U.T is:\n {U.T}')
restored_X, distortion = task_c(U, X)
print(f'The restored vectors are: \n{restored_X}')
print(f'The distortion by comparing the restored vectors to the original vectors is {distortion}')

