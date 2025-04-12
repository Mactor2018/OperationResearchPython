# This is the data class for LP, including A, b, c
# ! Python >= 3.9
import numpy as np
from enum import Enum
from enum import auto
import sys
if sys.version_info < (3, 9):
    raise Exception(f"{sys.version_info} is not supported, please upgrade to at least python==3.9")

class OperationError(Exception):
    pass

class Data:
    class REL(Enum):
        GEQ = -1
        EQ = 0
        LEQ = 1
    def __init__(self, A:np.ndarray, b:np.ndarray, c:np.ndarray):
        assert A.shape == (b.shape[0], c.shape[0]), f"The shape of given A, b and c don't match together: {A.shape} != ({b.shape[0]}, {c.shape[0]})"
        self.A, self.b, self.c, self.n, self.m = A, b, c, *A.shape
        
    # def __init__(self, A:np.ndarray, b:np.ndarray, c:np.ndarray, n:int, m:int):
    #     assert (n, m) == A.shape, f"The shape of A is {A.shape}, which doesn't match with the input ({n}, {m})."
    #     assert (n, m) == (b.shape[0], c.shape[0]), f"The shape of given A, b and c don't match together: {A.shape} != ({b.shape[0]}, {c.shape[0]})"
    #     self.A, self.b, self.c, self.n, self.m = A, b, c, n, m

    def add_variable(self, factor:int, Ar:np.ndarray, br:float):
        self.A = np.pad(self.A, ((0,1), (0,1)), mode='constant', constant_values=0) # (n, m) -> ((n+1), (m+1))
        self.A[-1] = np.append(Ar, factor)

        self.c = np.append(self.c, 0)
    
    def add_constraint(self, Ar:np.ndarray, rel:REL, br:float):
        if rel == Data.REL.EQ:
            # A:  (n, m) -> (n+1, m)
            self.A[-1] = Ar 
            self.b = np.append(self.b, br)
            self.n += 1
        else:
            # A:  (n, m) -> (n+1, m+1); C:  (m,) -> (m+1,)
            self.add_variable(rel.data, Ar, br)

    def list2constraints(self,A:np.ndarray, rel_list: list[REL], b:np.ndarray):
        n0, n1, n2 = A.shape[0], len(rel_list), b.shape[0]
        assert n0==n1 and n1==n2, "The input data don't match in size: {n0}, {n1}, {n2}"
        for i in range(n1):
            self.add_constraint(A[i], rel_list[i], b[i])

    def __str__(self):
        return f"A=\n{A}\nB=\n{b}\nc=\n{c}"
    
    def size(self):
        return self.A.shape
    
    def __len__(self):
        raise OperationError("`__len__` is not supported, please use `object.size()`")
if __name__== "__main__":
    A = np.array(
        [[4,2,3],
        [5,0,1]], dtype=np.float32
    )
    b = np.array([3,1],dtype=np.float32)
    c = np.array([2,1.0,-0.5])
    d = Data(A,b,c)
    