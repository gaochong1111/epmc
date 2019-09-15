'''
Created on 2019年8月1日

@author: chenyan
'''
from common import decompose_into_positive_operators,get_support,is_square, is_zero_array,\
    is_positive, array_equal, projector_join, get_orth_complement
import numpy as np
from scipy.constants.constants import epsilon_0
from sympy.codegen.fnodes import dimension

class SuperOperator:
    '''
    super operator
    '''
    
    def __init__(self, kraus=[]):
        '''
        Constructor
        kraus: np.matrix
        '''
        self._kraus = kraus
        if len(kraus) != 0:
            self._dimension = kraus[0].shape[0]
            for m in kraus:
                if len(m.shape) != 2 or m.shape[0] != self._dimension or m.shape[1] != self._dimension:
                    return 
        else:
            self._dimension = 0
        self._kraus = kraus
        
    @property
    def kraus(self):
        return self._kraus
    
    @kraus.setter
    def kraus(self, value):
        if len(value) != 0:
            self._dimension = value[0].shape[0]
            for m in value:
                if len(m.shape) != 2 or m.shape[0] != self._dimension or m.shape[1] != self._dimension:
                    return 
        else:
            self._dimension = 0
        self._kraus = value
    
    @property
    def dimension(self):
        return self._dimension
    
    @dimension.setter
    def dimension(self, value):
        self._dimension = value
    
    def get_matrix_representation(self):
        res = np.matrix(np.zeros(shape=[self.dimension ** 2, self.dimension ** 2],dtype=np.complex))
        for i in range(len(self.kraus)):
            res += np.kron(self.kraus[i], np.conjugate(self.kraus[i]))
        return res
    
    def get_positive_eigen_operators(self):
        '''
        Get a (complete, but not necessarily linear independent) set of 
        positive fixed-operators for the super-operator 
        '''
        res = []
        if self.dimension == 0 or len(self.kraus) == 0:
            return res

        matrix_representation = self.get_matrix_representation()
        
        eigen_values, eigen_vectors = np.linalg.eig(matrix_representation)
        
        for i in range(len(eigen_values)):
            if np.abs((eigen_values[i] - 1.0).real) < epsilon_0 and np.abs(eigen_values[i].imag) < epsilon_0:
                # eigen_vectors[:,i] shape (dimension, 1) is a matrix not a vector
                eigen_matrix = np.matrix(eigen_vectors[:,i].reshape([self.dimension, self.dimension]))
                
                real_positive, real_negative, image_positive, image_negative = decompose_into_positive_operators(eigen_matrix)
                
                if not is_zero_array(real_positive):
                    res.append(real_positive)
                if not is_zero_array(real_negative):
                    res.append(real_negative)
                if not is_zero_array(image_positive):
                    res.append(image_positive)
                if not is_zero_array(image_negative):
                    res.append(image_negative)
                    
        return res
    
    def apply_on_operator(self, operator):
        if not is_square(operator):
            return     
        if not self.dimension == operator.shape[0]:
            return
        
        res = np.matrix(np.zeros([self.dimension, self.dimension], dtype=np.complex))
        
        for i in range(len(self.kraus)):
            res += self.kraus[i] * operator * self.kraus[i].H
        
        return res
    
    def product_super_operator(self, super_operator):
        new_matrix = self.get_matrix_representation() * super_operator.get_matrix_representation()
        return create_from_matrix_representation(new_matrix)
    
    def product_operator(self, operator):
        kraus = []
        kraus.append(operator)
        super_operator = SuperOperator(kraus)
        return super_operator.product_super_operator(self)
    
    def power(self, n):
        if n == 0:
            karus = []
            karus.append(np.eye(self.dimension, self.dimension, dtype=np.complex))
            return SuperOperator(karus)
        elif n > 0:
            return self.product_super_operator(self.power(n - 1))
        
    def max_period(self):
        projects = self.bscc_decomposition()  
        periods = np.zeros(len(projects) - 1, dtype=np.int32)
        
        for i in range(1, len(projects)):
            periods[i - 1] = len(self.period_decomposition(projects[i]))
        
        return np.max(periods)
    
    def infinity(self):
        this_matrix = self.get_matrix_representation()
        start = 15
        start_matrix = self.get_matrix_representation()
        
        for _ in range(start):
            start_matrix = start_matrix * start_matrix
        
        max_period = self.max_period()
        for _ in range(max_period):
            start_matrix = start_matrix + start_matrix * this_matrix
        
        start_matrix = 1.0 / max_period * start_matrix
        previous_matrix = start_matrix
        
        n = 1
        while True:
            current_matrix =  n * previous_matrix * this_matrix + start_matrix
            current_matrix = 1.0 / (n + 1) * current_matrix
            if is_zero_array(previous_matrix - current_matrix):
                break
            previous_matrix = current_matrix
            ++n
        return create_from_matrix_representation(current_matrix)
    
    def check_bscc(self, projector):
        support = get_support(self.apply_on_operator(projector))
        if not is_positive(projector - support):
            return False
        
        matrix_representation = np.kron(projector, np.conjugate(projector)) * self.get_matrix_representation()
        
        pro_so = create_from_matrix_representation(matrix_representation)
        fix_points = pro_so.get_positive_eigen_operators()
        
        if len(fix_points) != 1:
            return False
        
        return array_equal(fix_points[0], projector)

    def get_bscc(self, projector):
        matrix_representation = np.kron(projector, np.conjugate(projector)) * self.get_matrix_representation()
        
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
            
            res.extend(self.get_bscc(projector_s))
            res.extend(self.get_bscc(complement))
            return res
    
    def bscc_decomposition(self):
        dimension = self.dimension
        res = self.get_bscc(np.eye(dimension, dimension, dtype=np.complex))
        
        stationary = np.matrix(np.zeros([dimension, dimension], dtype=np.complex))
        for projector in res:
            stationary = projector_join(stationary, projector)
        
        res.insert(0, get_orth_complement(stationary))
        
        return res
        
    def period_decomposition(self, bscc):
        res = []
        
        if not self.check_bscc(bscc):
            return res
        
        so = self.product_operator(bscc)
        matrix_representation = so.get_matrix_representation()
        
        eigen_values, _ = np.linalg.eig(matrix_representation)
        
        period = 0
        
        for i in range(len(eigen_values)):
            if np.abs((eigen_values[i] - 1.0).real) < epsilon_0 and np.abs(eigen_values[i].imag) < epsilon_0:
                period = period + 1
        
        return self.power(period).get_bscc(bscc)   
        
        
        
def create_from_choi_representation(matrix):
    if not is_square(matrix):
        print("Matrix is not a square!")
        return
    dimension = matrix.shape[0]
    sqrt = np.sqrt(dimension)
    
    if sqrt - np.floor(sqrt) > epsilon_0:
        print("Sqrt is not a integer!")
        return
    
    n_dimension = int(sqrt)
    
    kraus = []
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    
    for i in range(len(eigen_values)):
        if eigen_values[i].real > epsilon_0:
            eigen_vector = eigen_vectors[:,i] * np.sqrt(eigen_values[i].real)
            if not is_zero_array(eigen_vector):
                kraus.append(eigen_vector.reshape([n_dimension, n_dimension]))
    
    return SuperOperator(kraus)

def create_from_matrix_representation(matrix):
    if not is_square(matrix):
        print("Matrix is not a square!")
        return
    dimension = matrix.shape[0]
    sqrt = np.sqrt(dimension)
    
    if sqrt - np.floor(sqrt) > epsilon_0:
        print("Sqrt is not a integer!")
        return
    
    n_dimension = int(sqrt)
    choi_matrix = np.matrix(np.zeros([dimension, dimension], dtype=np.complex))
    
    for k in range(n_dimension):
        for n in range(n_dimension):
            for m in range(n_dimension):
                for j in range(n_dimension):
                    choi_matrix[k * n_dimension + m, n * n_dimension +j] = matrix[k * n_dimension + n, m * n_dimension +j]
    
    return create_from_choi_representation(choi_matrix)


if __name__ == '__main__':
    m = np.zeros(shape=[4, 4], dtype=np.complex)
    m[0, 0] = 1.0 + 0.0j
    m[1, 1] = 1.0 / np.sqrt(2) + 0.0j
    m[2, 2] = 1.0 / np.sqrt(2) + 0.0j
    m[0, 3] = 0.5 + 0.0j
    m[3, 3] = 0.5 + 0.0j
    m = np.matrix(m)
    so = create_from_matrix_representation(m)
    so_kraus = so.kraus
    for x in so_kraus:
        print(x)


    


            
        
        