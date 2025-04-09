import numpy as np 
""" This is a scratchpad for LP
Standard form:
Maximize: c^T @ x
Subject to: A @ x <= b
x >= 0

c : n x 1 vector
A : m x n matrix
b : m x 1 vector
x : n x 1 vector

step1: input n,m and c, A, b
step2: find if there's identity matrix (vectors). if not, add artificial variables. Select the basis
step3: calculate the deltas = c_j - C_B ^ T @ A_j
step4: if not optimal yet, select the k-th and calculate theta = b_i / a_ij, choose the smallest one, x_r in while x_k out
step5: Change the basis and let A[i=r] = A[i=r]/=a_rk, and make A[j=k] be a identity vector.
step6: Go to step 3
step7: End with outputing x* and z*
"""
def find_identity_matrix_columns(A, tol=1e-8):
    """
    在矩阵 A 的列中寻找恰好组成单位矩阵的列，即返回一个长度为 n 的列表，
    列表中第 i 个元素为与标准基向量 e_i 匹配的列索引；
    若找不到全部匹配，则返回当前匹配结果（其中可能有 None）。
    """
    n, m = A.shape
    identity = np.eye(n)
    used = [False] * n
    match_indices = [None] * n

    for col_idx in range(m):
        col = A[:, col_idx]
        for i in range(n):
            if not used[i] and np.allclose(col, identity[:, i], atol=tol):
                match_indices[i] = col_idx
                used[i] = True
                break
        if all(used):
            return match_indices
    return match_indices


def main(A_data:list, b_data:list, c_data:list):
    # --------------------------
    # 输入问题：标准形 LP
    # 最大化： c^T x
    # 约束： A x <= b, x >= 0
    # --------------------------
    # 示例数据
    # A_data = [
    #     [0, 5, 1, 0, 0],
    #     [6, 2, 1, 1, 0],
    #     [1, 1, 0, 0, 1]
    # ]
    # b_data = [15, 24, 5]
    # c_data = [2, 1, 0, 0, 0]

    n = len(A_data)      # 行数
    m = len(A_data[0])   # 列数

    # 转换为 numpy 数组（采用 float16 可修改为 float64 以提高精度）
    BIG_M = 1000 * np.max(A_data)
    convert_func = lambda y: np.array(y, dtype=np.float64)
    A = convert_func(A_data)
    b = convert_func(b_data)
    c = convert_func(c_data)

    # 试找是否有单位向量列作为初始基
    ivIndicies = find_identity_matrix_columns(A)

    print("初始基变量索引:", ivIndicies)

    # 如果未找到完整的初始基，则引入人工变量（使用 Big M 法）
    if None in ivIndicies:
        # 添加人工变量：在 A 后面拼接单位阵，同时更新 c（目标系数为 -BIG_M）
        A = np.concatenate((A, np.eye(n, dtype=np.float64)), axis=1)
        c = np.concatenate((c, -np.ones(n, dtype=np.float64) * BIG_M), axis=0)
        ivIndicies = [idx for idx in range(m, m+n)]
        m = m + n  # 更新 A 的列数

    # 构造增强系数矩阵 Aug = [A | b] ，b重塑为列向量
    Aug = np.concatenate((A, b.reshape(-1, 1)), axis=1)

    # 设置最大迭代次数，防止死循环
    max_iter = 50
    iter_count = 0

    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("达到最大迭代次数，可能存在循环或退化问题。")
            break

        # 当前基变量对应的目标系数向量 C_B
        C_B = np.array([ c[i] for i in ivIndicies ])
        
        # 计算 deltas = c_j - C_B * A[:,j] 对于 j=0,..., m-1
        deltas = np.zeros(m)
        for j in range(m):
            # 这里 A[:, j] 为 j 列， C_B 为一维数组，二者点积即为 sum(C_B * A[:, j])
            deltas[j] = c[j] - np.dot(C_B, A[:, j])
        
        # 如果所有 deltas <= 0，则达到最优（注意对于最大化问题，进入规则选取正值）
        k0 = np.where(deltas > 0)[0]
        if k0.size == 0:
            # 构造最优解：对每个变量，若在基中则取对应 b 的值，否则为 0
            X = np.zeros(m)
            # 但注意：原始变量数不超过初始 m_data，人工变量通常忽略
            for idx in ivIndicies:
                X[idx] = b[ ivIndicies.index(idx) ]  # 此处简单起见：对应于 b 的值
            print("最优解 (基变量):")
            print("基变量索引:", ivIndicies)
            # 输出目标函数值（基于 C_B 与 b）
            z = np.dot(C_B, b)
            print("最优目标值 Z* =", z)
            break

        # 选择第一个正增益的非基变量作为进入变量（你可以改用 Bland 规则等更稳定的规则）
        k = k0[0]

        # 对于每一行，计算 theta = b_i / Aug[i,k]，仅当 Aug[i,k] > 0，否则设为一个大数（BIG_M）
        thetas = np.full(n, BIG_M, dtype=np.float64)
        for i in range(n):
            if Aug[i, k] > 1e-8:  # 只考虑正 pivot 元
                thetas[i] = Aug[i, -1] / Aug[i, k]

        # 选择最小正 theta 对应的行（离基变量中要出基的那个变量）
        r0 = np.argmin(thetas)
        leaving_var = ivIndicies[r0]

        print(f"迭代 {iter_count}: x_{k} 进入， x_{leaving_var} 离出基")
        
        # 进行主元变换：首先归一化主元行
        pivot = Aug[r0, k]
        Aug[r0] = Aug[r0] / pivot

        # 其余行消去
        for i in range(n):
            if i != r0:
                factor = Aug[i, k]
                Aug[i] = Aug[i] - factor * Aug[r0]

        # 更新基变量索引
        ivIndicies[r0] = k

        # 同步更新 A 和 b（A 为 Aug 前 m 列，b 为最后一列）
        A = Aug[:, :m]
        b = Aug[:, -1]

        # 输出当前增强矩阵（调试用）
        # print("当前增强矩阵 Aug:")
        # print(Aug)
        
if __name__ == "__main__":
    # 示例：约束为 x1 + x3 = 100, x2 + x4 = 120, 目标函数 maximize 200 x1 + 300 x2
    A_data = [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 2, 0, 0, 1]
    ]
    b_data = [100, 120, 160]
    c_data = [200, 300, 0, 0, 0]
    main(A_data, b_data, c_data)