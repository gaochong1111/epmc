'''
Created on 2019年8月1日

@author: chenyan
'''
import numpy as np
np.set_printoptions(threshold=np.inf)
from common import is_positive, get_support, decompose_into_positive_operators,\
    is_zero_array, projector_join, get_orth_complement
from common import array_equal,matrix_infinite
from super_operator import create_from_matrix_representation, SuperOperator
from scipy.linalg.decomp_svd import orth
from scipy.constants.constants import epsilon_0

'''
states: int set
Q: qmc (int, int): superoperator
pri: state priority int: int
classical_state: int
super_operator_demension: int
'''
def pqmc_values(states, Q, pri, classical_state, super_operator_demension):
    if classical_state > np.max(states):
        print('The classical state index out of range!')
        return
    
    state_demension = np.max(states) + 1
    I_c = np.eye(state_demension, state_demension, dtype = np.complex)
    I_H = np.eye(super_operator_demension, super_operator_demension, dtype=np.complex)
    
    E_s = np.kron(I_c[classical_state].reshape([state_demension, 1]), I_H)
    M_s = np.kron(E_s, np.conjugate(E_s))
    print("M_s:")
    print(M_s.shape)
    
    M_c = np.zeros([super_operator_demension ** 2, (super_operator_demension ** 2) * (state_demension ** 2)], dtype=np.complex)
    for s in states:
        E = np.kron(I_c[s].reshape([1, state_demension]), I_H)
        M_c += np.kron(E, np.conjugate(E))
    print("M_c:")
    print(M_c.shape)
    
    M = np.zeros([(super_operator_demension ** 2) * (state_demension ** 2), (super_operator_demension ** 2) * (state_demension ** 2)], dtype=np.complex)
    for key, value in Q.items():
        E_i_kraus = value.kraus
        adjacency_matrix = np.zeros([state_demension, state_demension], dtype=np.complex)
        adjacency_matrix[key[1], key[0]] = 1.0
        for e in E_i_kraus:
            E = np.kron(adjacency_matrix, e)
            M += np.kron(E, np.conjugate(E))
    print("M:")
    print(M.shape)
    
    M_infinite = matrix_infinite(M)
    print("M_infinite:")
    print(M_infinite)
    
    M_even = np.zeros([(super_operator_demension ** 2) * (state_demension ** 2), (super_operator_demension ** 2) * (state_demension ** 2)], dtype=np.complex)
    bscc_min_pri = dict()
    B = get_bscc(M, np.kron(I_c, I_H))
    print("B:")
    print(B)
    for b in B:
        C_b = []
        vecs = orth(b)
        for vec in vecs:
            nonzero_index = -1
            flag = False
            for i in range(state_demension):
                if not is_zero_array(vec[i * super_operator_demension:(1 + i) * super_operator_demension]):
                    if not flag:
                        flag = True
                        nonzero_index = i
                    else:
                        flag = False
                        break
            if flag:
                C_b.append(nonzero_index)
        bscc_min_pri[b] = np.min(C_b)
    EP = set()
    for key, value in pri:
        if value % 2 == 0:
            EP.add(key)
    for k in EP:
        P_k = np.zeros([super_operator_demension * state_demension, super_operator_demension * state_demension], dtype=np.complex)
        for key, value in bscc_min_pri:
            if value == k:
                P_k += key
        M_even += np.kron(P_k, P_k)
    
    return M_c * M_even * M_infinite * M_s 

def check_bscc(super_operator, projector):
    '''
    '''
    support = get_support(super_operator.apply_on_operator(projector))
    if not is_positive(projector - support):
        return False
    
    matrix_representation = np.kron(projector, np.conjugate(projector)) * super_operator.get_matrix_representation()
    
    pro_so = create_from_matrix_representation(matrix_representation)
    fix_points = pro_so.get_positive_eigen_operators()
    
    if len(fix_points) != 1:
        return False
    
    return array_equal(fix_points[0], projector)


def get_bscc(super_operator, projector):
    matrix_representation = np.kron(projector, np.conjugate(projector)) * super_operator.get_matrix_representation()
    
    pro_so = create_from_matrix_representation(matrix_representation)
    
    fix_points = pro_so.get_positive_eigen_operators()
    
    '''fix_points[i] compare to projector'''
    
    fix_points = sorted(fix_points, key=lambda x: np.linalg.matrix_rank(x))
    pop_index = []
    
    for i in range(len(fix_points) - 1):
        support_i = get_support(fix_points[i])
        for j in range(i + 1, len(fix_points)):
            if j in pop_index:
                continue
            support_j = get_support(fix_points[j])
            if array_equal(fix_points[i], fix_points[j]) or is_positive(support_j - support_i):
                pop_index.append(j)
    
    for index in pop_index:
        fix_points.pop(index)
    
    
    res = []
    
    if len(fix_points) == 0:
        return res
    elif len(fix_points) == 1:
        res.append(get_support(fix_points[0]))
        return res
    else:
        diff = fix_points[0] - fix_points[1]
        real_positive, real_negative, _, _ = decompose_into_positive_operators(diff)
        projector_s = None
        if is_zero_array(real_positive):
            projector_s = get_support(real_negative)
        else:
            projector_s = get_support(real_positive)
        complement = projector - projector_s
        
        res.extend(get_bscc(super_operator, projector_s))
        res.extend(get_bscc(super_operator, complement))
        return res
    
def bscc_decomposition(super_operator):
    dimension = super_operator.dimension
    res = get_bscc(super_operator, np.eye(dimension, dimension, dtype=np.complex))
    
    stationary = np.matrix(np.zeros([dimension, dimension], dtype=np.complex))
    for projector in res:
        stationary = projector_join(stationary, projector)
    
    res.insert(0, get_orth_complement(stationary))
    
    return res
    
def period_decomposition(super_operator, bscc):
    res = []
    
    if not check_bscc(super_operator, bscc):
        return res
    
    so = super_operator.apply_on_operator(bscc)
    matrix_representation = so.get_matrix_representation()
    
    eigen_values, _ = np.linalg.eig(matrix_representation)
    
    period = 0
    
    for i in range(len(eigen_values)):
        if np.abs((eigen_values[i] - 1.0).real) < epsilon_0 and np.abs(eigen_values[i].imag) < epsilon_0:
            period = period + 1
    
    return get_bscc(super_operator.power(period), bscc)
        
        

if __name__ == '__main__':
    states = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    Q = {(2,4):[[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j]],(2,6):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j]],(6,6):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j]],(0,3):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.5000000+0.0000000j],[0.0000000+0.0000000j,0.7071068+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.7071068+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.5000000+0.0000000j]],(4,3):[[ 0.5000000+0.0000000j, 0.5000000+0.0000000j, 0.5000000+0.0000000j, 0.5000000+0.0000000j],[ 0.5000000+0.0000000j,-0.5000000+0.0000000j, 0.5000000+0.0000000j,-0.5000000+0.0000000j],[ 0.5000000+0.0000000j, 0.5000000+0.0000000j,-0.5000000+0.0000000j,-0.5000000+0.0000000j],[ 0.5000000+0.0000000j,-0.5000000+0.0000000j,-0.5000000+0.0000000j, 0.5000000+0.0000000j]],(1,3):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.5000000+0.0000000j],[0.0000000+0.0000000j,0.7071068+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.7071068+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.5000000+0.0000000j]],(3,5):[[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j]],(3,7):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j]],(5,3):[[ 0.5000000+0.0000000j, 0.5000000+0.0000000j, 0.5000000+0.0000000j, 0.5000000+0.0000000j],[ 0.5000000+0.0000000j,-0.5000000+0.0000000j, 0.5000000+0.0000000j,-0.5000000+0.0000000j],[ 0.5000000+0.0000000j, 0.5000000+0.0000000j,-0.5000000+0.0000000j,-0.5000000+0.0000000j],[ 0.5000000+0.0000000j,-0.5000000+0.0000000j,-0.5000000+0.0000000j, 0.5000000+0.0000000j]],(7,7):[[1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,1.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j,0.0000000+0.0000000j],[0.0000000+0.0000000j,0.0000000+0.0000000j,0.0000000+0.0000000j,1.0000000+0.0000000j]]}    
    pri = {0:1,1:0,2:1,3:0,4:1,5:0,6:1,7:0}
    classical_state = 0
    super_operator_demension = 2
    Q_prim = dict()
    
    for key, value in Q.items():
        Q_prim[key] = create_from_matrix_representation(np.array(value))
    
    Q = Q_prim
        
    print(pqmc_values(states, Q, pri, classical_state, super_operator_demension))
    '''
    m = np.matrix(np.zeros([2, 2], dtype = np.complex))
    m[0, 0] = 1.0 + 0.0j
    m[1, 1] = -1.0 + 0.0j
    a = np.array([[5 + 1.0j, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]], dtype = np.complex)
    m1 = Matrix(a)
    P, J = m1.jordan_form()
    print(J)
    c = np.matrix(J).astype(np.complex)
    print(jordan_eigen_value(c))
    print(matrix_infinite(a))
    
    so = SuperOperator([m])
    projector = np.matrix(np.eye(2, 2, dtype = np.complex))
    print(get_bscc(so, projector))
    
    for p in get_bscc(so, projector):
        print(orth(p))
    '''
    
