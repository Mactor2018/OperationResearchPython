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
        A, b, c, rel_list = args
        if len(args) == 1: # type c
            self.data = args[0]
        elif len(args) == 4: 
            if isinstance( A, list ): # type a
                A,b,c = np.array(A, dtype=np.float32),np.array(b, dtype=np.float32),np.array(c, dtype=np.float32)

            elif isinstance(A, np.ndarray): # type b
                pass
                # self.data = Data(A, b, c)
            # self.data = Data(A,b,c)
            self.data = Data(c=c)
            self.data.list2constraints(A, rel_list, b)
        else:
            raise InputError(f"""Please initiate the instance following the format:\nParams:
            a. (obj:OBJ, A: list[list], b:list, c:list, rel_list:list[Data.REL])
            b. (obj:OBJ, A:np.ndarray,  b:np.ndarray, c:np.ndarray, rel_list: list[Data.REL])
            c. (obj:OBJ, LPdata:Data)""")
        
        # Standardize the objective method:
        self.data.c *= obj.value 
        
    def find_identity_matrix_columns(self, tol=1e-8):
        """
        在矩阵 A 的列中寻找恰好组成单位矩阵的列，即返回一个长度为 n 的列表，
        列表中第 i 个元素为与标准基向量 e_i 匹配的列索引；
        若找不到全部匹配，则返回当前匹配结果（其中可能有 None）。
        """
        n, m = self.data.A.shape
        identity = np.eye(n)
        used = [False] * n
        match_indices = [None] * n

        for col_idx in range(m):
            col = self.data.A[:, col_idx]
            for i in range(n):
                if not used[i] and np.allclose(col, identity[:, i], atol=tol):
                    match_indices[i] = col_idx
                    used[i] = True
                    break
            if all(used):
                return match_indices
        return match_indices
    
    def solve(self) -> dict:
        """
        利用单纯形法（Big M 法）求解标准形线性规划：
        最大化： c^T x
        约束： A x <= b, x >= 0
        使用 self.data 中的 A, b, c（均为 float32），且在初始化时已调整目标函数符号。

        返回字典：
        当问题不可行时，返回 {"feasible": False}
        当存在最优解时，返回 {"feasible": True, "X_star": <原始变量最优解>, "z_star": <最优目标值>}
        """
        # 使用数据拷贝，保持 float32 类型
        A = self.data.A.copy()
        b = self.data.b.copy()
        c = self.data.c.copy()
        n, m = self.data.size()

        # 设定 BIG_M（根据 A 中元素的绝对值最大值）
        BIG_M = 1000 * np.max(np.abs(A))
        tol = 1e-8

        # 尝试直接找到初始基（单位矩阵列），无需额外转换
        ivIndices = self.find_identity_matrix_columns()
        artificial_offset = m  # 原始变量的数量
        if None in ivIndices:
            # 拼接单位阵到 A 的右侧引入人工变量
            A = np.concatenate((A, np.eye(n, dtype=np.float32)), axis=1)
            # 对应人工变量的目标系数设为 -BIG_M（保持 float32）
            c = np.concatenate((c, -np.ones(n, dtype=np.float32) * BIG_M), axis=0)
            # 此时初始基变量全部为人工变量
            ivIndices = [idx for idx in range(m, m + n)]
            m += n

        # 构造增强矩阵 Aug = [A | b]
        Aug = np.concatenate((A, b.reshape(-1, 1)), axis=1)

        # 设置最大迭代次数，防止死循环
        max_iter = 50
        iter_count = 0

        while True:
            iter_count += 1
            if iter_count > max_iter:
                return {"feasible": False}

            # 采用向量化取当前基变量对应的目标系数：C_B = c[ivIndices]
            C_B = np.take(c, ivIndices)
            # 计算每个变量的增广费用：deltas = c - (C_B dot A)；注意 np.dot(C_B, A) 得到 (m,) 的向量
            deltas = c - np.dot(C_B, A)

            # 若所有增广费用均不大于 tol，则已达到最优
            if np.all(deltas <= tol):
                # 若基中存在人工变量，检查其对应 b 值是否为 0，若不为 0 则问题不可行
                infeasible = any(
                    abs(b[ivIndices.index(idx)]) > tol 
                    for idx in ivIndices if idx >= artificial_offset
                )
                if infeasible:
                    return {"feasible": False}
                else:
                    # 构造最优解，返回原始变量部分
                    X = np.zeros(m, dtype=np.float32)
                    for row, idx in enumerate(ivIndices):
                        X[idx] = b[row]
                    z = np.dot(C_B, b)

                    self.data_updated = Data(A,b,c)
                    return {"feasible": True, "X_star": X[:artificial_offset], "z_star": z, "data_updated": self.data_updated}

            # 选择第一个正增广费用的变量作为进入变量
            candidate = np.where(deltas > tol)[0]
            k = candidate[0]

            # 计算每一行的 theta 值：若 Aug[i,k] > tol 则 theta = b_i / Aug[i,k]，否则赋值 BIG_M
            thetas = np.full(n, BIG_M, dtype=np.float32)
            for i in range(n):
                if Aug[i, k] > tol:
                    thetas[i] = Aug[i, -1] / Aug[i, k]

            # 若找不到合法的 pivot（所有 theta 均为 BIG_M），则问题不可行或无界
            if np.min(thetas) == BIG_M:
                return {"feasible": False}

            # 选择 theta 值最小的行作为离基变量
            r0 = int(np.argmin(thetas))
            pivot = Aug[r0, k]
            Aug[r0] /= pivot  # 主元行归一化

            # 对其他行进行消元
            for i in range(n):
                if i != r0:
                    Aug[i] -= Aug[i, k] * Aug[r0]
                    
            # 更新基变量，将 r0 行对应的基变量替换为新进入变量 k
            ivIndices[r0] = k

            # 同步更新 A 与 b（Aug 的前 m 列和最后一列）
            A = Aug[:, :m]
            b = Aug[:, -1]


if __name__=="__main__":
    # 示例：输入矛盾约束（无可行解），例如：
    # x1 + x3 = 100, x1 - x2 = 200，可能构造出矛盾
    # A_data = [
    #     [1, 0, 1],
    #     [1, -1, 0]
    # ]
    # b_data = [1400, 200]
    # c_data = [2, 1, 0]
    # rel_list = [Data.REL.GEQ, Data.REL.LEQ]
    A_data = [
        [1,0],
        [0,1],
        [1,2]
    ]
    b_data = [100, 120, 160]
    c_data = [-200,-300]

    rel_list = [Data.REL.LEQ, Data.REL.LEQ, Data.REL.LEQ]
    lp = LinearProgram(LinearProgram.OBJ.MIN, A_data, b_data, c_data, rel_list)
    print( lp.solve() )