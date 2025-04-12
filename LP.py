import numpy as np
from enum import Enum
from enum import auto
from data import Data 

class InputError(Exception):
    pass

class LinearProgram:
    class OBJ(Enum):
        MAX = 1
        MIN = -1
    

    def __init__(self, obj:OBJ, *args):
        """
        Params:
        a. (obj:OBJ, A: list[list], b:list, c:list, rel_list:list[Data.REL])
        b. (obj:OBJ, A:np.ndarray,  b:np.ndarray, c:np.ndarray, rel_list: list[Data.REL])
        c. (obj:OBJ, LPdata:Data)
        """
        if len(args) == 1: # type c
            self.data = args.LPdata
        elif len(args) == 4: 
            if isinstance(args.A, list): # type a
                self.data = Data(
                    np.array(args.A, dtype=np.float32),
                    np.array(args.b, dtype=np.float32),
                    np.array(args.c, dtype=np.float32)
                )
                # self.data.list2constraints(args.rel_list) 

            elif isinstance(args.A, np.ndarray): # type b
                self.data = Data(args.A, args.b, args.c)
            
            self.data.list2constraints(args.rel_list)
        else:
            raise InputError(f"""Please initiate the instance following the format:\nParams:
            a. (obj:OBJ, A: list[list], b:list, c:list, rel_list:list[Data.REL])
            b. (obj:OBJ, A:np.ndarray,  b:np.ndarray, c:np.ndarray, rel_list: list[Data.REL])
            c. (obj:OBJ, LPdata:Data)""")
        
        # Standardize the objective method:
        self.c *= obj.value 
        
    def find_identity_matrix_columns(self, tol=1e-8):
        """
        在矩阵 A 的列中寻找恰好组成单位矩阵的列，即返回一个长度为 n 的列表，
        列表中第 i 个元素为与标准基向量 e_i 匹配的列索引；
        若找不到全部匹配，则返回当前匹配结果（其中可能有 None）。
        """
        n, m = self.A.shape
        identity = np.eye(n)
        used = [False] * n
        match_indices = [None] * n

        for col_idx in range(m):
            col = self.A[:, col_idx]
            for i in range(n):
                if not used[i] and np.allclose(col, identity[:, i], atol=tol):
                    match_indices[i] = col_idx
                    used[i] = True
                    break
            if all(used):
                return match_indices
        return match_indices