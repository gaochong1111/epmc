'''
Created on 2019年8月1日

@author: chenyan
'''
import numpy as np
np.set_printoptions(threshold=np.inf)
from common import is_positive, get_support, decompose_into_positive_operators,\
    is_zero_array, projector_join, get_orth_complement
from common import array_equal
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
    
    super_operator_M = create_from_matrix_representation(M)
    M_infinite = super_operator_M.infinity().get_matrix_representation()
    print("M_infinite:")
    print(M_infinite)
    
    M_even = np.zeros([(super_operator_demension ** 2) * (state_demension ** 2), (super_operator_demension ** 2) * (state_demension ** 2)], dtype=np.complex)
    bscc_min_pri_key = []
    bscc_min_pri_value = []
    B = super_operator_M.get_bscc(np.kron(I_c, I_H))
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
        bscc_min_pri_key.append(b)
        bscc_min_pri_value.append(np.min(C_b))
    EP = set()
    print(pri)
    for key, value in pri.items():
        if value % 2 == 0:
            EP.add(key)
    for k in EP:
        P_k = np.zeros([super_operator_demension * state_demension, super_operator_demension * state_demension], dtype=np.complex)
        for i in range(len(bscc_min_pri_key)):
            if bscc_min_pri_value[i] == k:
                P_k += bscc_min_pri_key[i]
        M_even += np.kron(P_k, P_k)
    
    M_c = np.matrix(M_c)
    M_even = np.matrix(M_even)
    M_s = np.matrix(M_s)
    return M_c * M_even * M_infinite * M_s 
        

if __name__ == '__main__':
    '''
    identity = np.eye(16, 16, dtype = np.complex)
    so = create_from_matrix_representation(identity)
    print(so.infinity().get_matrix_representation())
    '''
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
    
    m = np.matrix(np.zeros([2, 2], dtype = np.complex))
    m[0, 1] = 1.0 + 0.0j
    m[1, 0] = 1.0 + 0.0j
    so = SuperOperator([m])
    projector = np.matrix(np.eye(2, 2, dtype = np.complex))
    print(so.get_bscc(projector))
    
    for p in so.get_bscc(projector):
        print(orth(p))
    '''
    
    
