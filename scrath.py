import numpy as np

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
    n = len(A_data)      # 行数
    m = len(A_data[0])   # 列数

    # 转换为 numpy 数组（采用 float64 以提高精度）
    BIG_M = 1000 * np.max(A_data)
    convert_func = lambda y: np.array(y, dtype=np.float64)
    A = convert_func(A_data)
    b = convert_func(b_data)
    c = convert_func(c_data)

    # 尝试寻找是否有单位向量列作为初始基
    ivIndicies = find_identity_matrix_columns(A)
    print("初始基变量索引:", ivIndicies)

    # 如果未找到完整初始基，则引入人工变量（使用 Big M 法）
    artificial_offset = m  # 原始变量数量 m 保留
    if None in ivIndicies:
        # 将单位阵拼接到 A 后面，同时更新目标系数 c（人工变量目标系数为 -BIG_M）
        A = np.concatenate((A, np.eye(n, dtype=np.float64)), axis=1)
        c = np.concatenate((c, -np.ones(n, dtype=np.float64) * BIG_M), axis=0)
        # 基变量全部为人工变量，新基索引为 [m, m+1, ..., m+n-1]
        ivIndicies = [idx for idx in range(m, m+n)]
        m = m + n  # 更新 A 的列数

    # 构造增强矩阵 Aug = [A | b]，其中 b 重塑为列向量
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
        
        # 计算增广费用（deltas）：delta_j = c_j - C_B dot A[:,j]，j=0,..., m-1
        deltas = np.zeros(m)
        for j in range(m):
            deltas[j] = c[j] - np.dot(C_B, A[:, j])
        
        # 如果所有 deltas <= 0，则当前解为最优
        if np.all(deltas <= 1e-8):
            # 若存在人工变量在基中，检查其对应 b 值是否为零
            infeasible = False
            for idx in ivIndicies:
                if idx >= artificial_offset:  # 人工变量
                    # 找到该基变量所在行
                    row = ivIndicies.index(idx)
                    if abs(b[row]) > 1e-6:
                        infeasible = True
                        break
            if infeasible:
                print("问题不可行（存在非零的人工变量）")
            else:
                # 构造最优解：对于每个变量，若在基中则取对应 b 的值，否则为 0
                X = np.zeros(m)
                for row, idx in enumerate(ivIndicies):
                    X[idx] = b[row]
                print("最优解 (基变量):")
                print("基变量索引:", ivIndicies)
                z = np.dot(C_B, b)
                print("最优目标值 Z* =", z)
            break

        # 选择第一个正增益变量作为进入变量
        candidate_indices = np.where(deltas > 1e-8)[0]
        k = candidate_indices[0]

        # 对于每一行，计算 theta = b_i / Aug[i,k]（仅当 Aug[i,k] > 1e-8，否则赋值 BIG_M）
        thetas = np.full(n, BIG_M, dtype=np.float64)
        for i in range(n):
            if Aug[i, k] > 1e-8:
                thetas[i] = Aug[i, -1] / Aug[i, k]

        # 如果所有的 pivot 元素均不正，则说明该变量无法改变当前解，可能问题不可行或无界
        min_theta = np.min(thetas)
        if min_theta == BIG_M:
            print(f"在第 {iter_count} 次迭代中，变量 x_{k} 作为进入变量时无合法离基变量，问题可能不可行。")
            break

        # 选择使 theta 最小的行作为离基变量
        r0 = np.argmin(thetas)
        leaving_var = ivIndicies[r0]
        print(f"迭代 {iter_count}: x_{k} 进入， x_{leaving_var} 离出基")
        
        # 主元变换：先对 pivot 行归一化
        pivot = Aug[r0, k]
        Aug[r0] = Aug[r0] / pivot

        # 对其它各行进行消元
        for i in range(n):
            if i != r0:
                factor = Aug[i, k]
                Aug[i] = Aug[i] - factor * Aug[r0]

        # 更新基变量索引
        ivIndicies[r0] = k

        # 同步更新 A 与 b：A 为 Aug 的前 m 列，b 为最后一列
        A = Aug[:, :m]
        b = Aug[:, -1]

if __name__ == "__main__":
    # 示例：输入矛盾约束（无可行解），例如：
    # x1 + x3 = 100, x1 - x2 = 200，可能构造出矛盾
    A_data = [
        [1, 0, 1],
        [1, -1, 0]
    ]
    b_data = [100, 200]
    c_data = [2, 1, 0]
    main(A_data, b_data, c_data)
