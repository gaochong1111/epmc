'''
Created on 2019年8月1日

@author: chenyan
'''
import numpy as np
from scipy.constants.constants import epsilon_0
from scipy.linalg import orth
from sympy import Matrix
from scipy.linalg import null_space

def is_square(matrix):
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]

def is_zero_array(a):
    zero_array = np.zeros(a.shape, dtype=np.complex)
    return array_equal(a, zero_array)

def array_equal(a1, a2):
    if a1.shape != a2.shape:
        return False
    diff = a1 - a2
    diff_real = np.abs(diff.real)
    diff_imag = np.abs(diff.imag)
    if len(np.argwhere(diff_real > epsilon_0)) != 0 or len(np.argwhere(diff_imag > epsilon_0)) != 0:
        return False
    return True

def complex_equal(c1, c2):
    diff = c1 - c2
    if np.abs(diff.real) > epsilon_0 or np.abs(diff.imag) > epsilon_0:
        return False
    return True

def is_jordan_block(matrix):
    if not is_square(matrix):
        return False
    row_index = matrix.shape[0] - 1
    for i in range(0, row_index):
        if not complex_equal(matrix[i, i], matrix[i + 1, i + 1]):
            return False
    for k in range(0, row_index):
        if not complex_equal(matrix[k, k + 1], 1.0 + 0.j):
            return False
    return True

def jordan_eigen_value(J):
    i = 0
    eigen_value_map = dict()
    while i < J.shape[1]:
        start = i
        for j in range(1, J.shape[1] - start + 1):
            if is_jordan_block(J[start:start + j, start:start + j]):
                eigen_value_map.setdefault(J[start, start], []).append((start, j))
                i = i + j
                break
    return eigen_value_map

def matrix_infinite(matrix):
    print("begin")
    P, J = Matrix(matrix).jordan_form()
    print("J:")
    print(J)
    J = np.matrix(J).astype(np.complex)
    P = np.matrix(P).astype(np.complex)
    eigen_value_map = jordan_eigen_value(J)
    res = np.zeros([J.shape[0], J.shape[1]], dtype=np.complex)
    for key, value in eigen_value_map.items():
        if complex_equal(key, 1.0 + 0.j):
            for item in value:
                jordan_block = np.zeros([J.shape[0], J.shape[1]], dtype=np.complex)
                for i in range(item[1]):
                    jordan_block[item[0] + i, item[0] + i] = key
                for j in range(item[1] - 1):
                    jordan_block[item[0] + j, item[0] + j + 1] = 1.0 + 0.j
                res += jordan_block
    return res
    
def check_std_ort(basis):
    #subspace is null
    if len(basis) == 0:
        return False
    
    c0 = np.complex(0.0, 0.0) # 0.0 + 0.0j
    c1 = np.complex(1.0, 0.0) # 1.0 + 0.0j
    dimension = len(basis[0]) # dimension of subspace
    index1_set = set() # 分量为1的下标集合
    
    for v in basis:
        if len(v) != dimension:
            return False
        index0 = np.argwhere(v == c0) # TODO 视具体情况决定是否使用delta
        index1 = np.argwhere(v == c1)
        if index1.shape[0] != 1 or index0.shape[0] != dimension - 1:
            return False
        else:
            index1 = index1[0][0]
            if index1 in index1_set:
                return False
            else:
                index1_set.add(index1)
    return True

def is_positive(operator):
    if not is_square(operator):
        return False
    
    if array_equal(operator, np.zeros(operator.shape, dtype=np.complex)):
        return True
    
    eigen_values, _ = np.linalg.eig(operator)
    
    for value in eigen_values:
        if value.real < -epsilon_0:
            return False
    
    return True
        

def decompose_into_positive_operators(operator):
    if not is_square(operator):
        return
    dimension = operator.shape[0]
    real_part = operator.real
    image_part = operator.imag
    real_positive = np.matrix(np.zeros(shape=[dimension, dimension], dtype=np.complex))
    real_negative = np.matrix(np.zeros(shape=[dimension, dimension], dtype=np.complex))
    image_positive = np.matrix(np.zeros(shape=[dimension, dimension], dtype=np.complex))
    image_negative = np.matrix(np.zeros(shape=[dimension, dimension], dtype=np.complex))
    
    eigen_values, eigen_vectors = np.linalg.eig(real_part)

    for i in range(len(eigen_values)):
        if eigen_values[i] > epsilon_0:
            real_positive += eigen_values[i] * np.outer(eigen_vectors[:,i], np.conjugate(eigen_vectors[:,i]))
        elif eigen_values[i] < -epsilon_0:
            real_negative -= eigen_values[i] * np.outer(eigen_vectors[:,i], np.conjugate(eigen_vectors[:,i]))
    
    eigen_values, eigen_vectors = np.linalg.eig(image_part)
    for i in range(len(eigen_values)):
        if eigen_values[i] > epsilon_0:
            image_positive += eigen_values[i] * np.outer(eigen_vectors[:,i], np.conjugate(eigen_vectors[:,i]))
        elif eigen_values[i] < -epsilon_0:
            image_negative -= eigen_values[i] * np.outer(eigen_vectors[:,i], np.conjugate(eigen_vectors[:,i]))
    
    return real_positive, real_negative, image_positive, image_negative

def get_support(operator):
    if not is_square(operator):
        return
    dimension = operator.shape[0]
    
    orth_basis = np.matrix(orth(operator))
    
    res = np.matrix(np.zeros([dimension, dimension], dtype=np.complex))
    for i in range(orth_basis.shape[1]):
        m = operator * orth_basis[:,i]
        if not is_zero_array(m):
            res += np.outer(orth_basis[:,i], np.conjugate(orth_basis[:,i]))

    return res

def projector_join(projector1, projector2):
    return get_support(projector1 + projector2)

def get_orth_complement(projector):
    if not is_square(projector):
        return
    dimension = projector.shape[0]
    
    res = np.matrix(np.zeros([dimension, dimension], dtype=np.complex))
    basis = np.matrix(null_space(projector))
    for i in range(basis.shape[1]):
        if is_zero_array(projector * basis[:, i]):
            res += np.matrix(np.outer(basis[:, i], np.conjugate(basis[:, i])))
    
    return res

if __name__ == '__main__':
    a = np.matrix(np.eye(3, 3, dtype=np.complex))
    a[0, 0] = 0
    a[1, 1] = 0
    basis = null_space(a)
    print(basis[:, 0].shape)
    print(get_orth_complement(a))
    