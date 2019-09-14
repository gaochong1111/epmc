'''
Created on 2019年7月26日

@author: chenyan
'''
import numpy as np
from common import *
import cmath
from scipy.constants.constants import epsilon_0
from scipy.linalg import orth

delta = 0.000001 #精度需要测试


def get_dimension(basis):
    if len(basis) == 0:
        return -1
    return len(basis[0])

def frobenius_norm(matrix):
    return np.sqrt(np.trace(matrix.H * matrix))


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


u1 = np.array([0.+0.j, 0.+0.j, 1.0+0.j, 0.+0.j], dtype=np.complex)
u2 = np.array([0.+0.j, 1.+0.j, 0.0+0.j, 0.+0.j], dtype=np.complex)
u3 = np.array([1.+0.j, 0.+0.j, 0.0+0.j, 0.+0.j], dtype=np.complex)
superoperator = np.eye(16, 16, dtype=np.complex)


c = np.matrix([[1 + 0.j, 0 + 0.j], [1 + 0.j, 0 + 0.j]], dtype=np.complex)
v1 = np.array([0.95711108+0.j,  0.26905317+0.10746522j], dtype=np.complex)
v2 = np.array([ 0.73730995+0.j, -0.66099414-0.13950191j], dtype=np.complex)
v3 = np.array([0.95711108-0.j,  0.26905317-0.10746522j], dtype=np.complex)
v4 = np.array([ 0.73730995-0.j, -0.66099414+0.13950191j], dtype=np.complex)
print(np.array(c).flatten('F'))
print(np.tensordot(np.array(c), np.array(c), 0))
print(c != 1 + 1j)
print(np.vdot(v1, v1))
print(np.inner(v3, v2))
print(c.imag)
print(c.real)
print(np.zeros(shape=[5, 5], dtype=np.complex))
print(decompose_into_positive_operators(c))
#check_bscc(c, [[1 + 1j, 2 + 3j], [1j, -1j]])
#print(c.T) # 转置
#print(c.H) # 转置共轭
#print(c.I) # 逆
values, vectors = np.linalg.eig(c)
print(vectors[:,0].shape)
print(np.linalg.eig(c))
print(orth(c))
print(np.outer(orth(c),orth(c)))
print(np.linalg.matrix_rank(c))
a = np.eye(4)
a1 = np.eye(4)
a1[1][1] = 0
a2 = np.eye(4)
a2[1][1] = 0
a2[2][2] = 0
l = [a,a1,a2]
print(sorted(l, key=lambda x: np.linalg.matrix_rank(x)))
l.pop(1)
print(l)