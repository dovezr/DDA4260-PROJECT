import numpy as np
import projectLib as lib
import math


# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

k = 50  # 这个是选取临近几个的参数，由于总共100个电影，初始化我先选定了相似度最高的50个

# print(R)


def construct_R_tilde(R, R_hat):  # 这里的R_hat是经过rbm预测以后的predictedRatings，R是原有的矩阵
    R_tilde = np.full((300, 100), np.nan)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if not np.isnan(R[i, j]):
                R_tilde[i, j] = R[i, j] - R_hat[i, j]

    return R_tilde


def cal_d(i, j):
    up = 0
    down_i = 0
    down_j = 0
    for t in range(len(i)):
        if np.isnan(i[t]) == False and np.isnan(j[t]) == False:
            up += i[t] * j[t]
            down_i += i[t] * i[t]
            down_j += j[t] * j[t]
    return up / (math.sqrt(down_i) * math.sqrt(down_j))


def order_list(li):
    sorted_values_with_index = sorted(
        enumerate(li), key=lambda x: abs(x[1]), reverse=True
    )

    # 提取排序后的值和对应的索引
    sorted_values = [value for index, value in sorted_values_with_index]
    original_indices = [index for index, value in sorted_values_with_index]
    return sorted_values, original_indices


def main(R_hat):  # 主要是运用这个函数达到目的
    R = np.full((300, 100), np.nan)
    for row in training:
        R[row[1], row[0]] = row[2]
    R_tilde = construct_R_tilde(R, R_hat)  # 缺失值用np.nan来表示

    for m in range(R_tilde.shape[1]):
        d_list = []
        for n in range(R_tilde.shape[1]):
            if m != n:
                d_list.append(cal_d(R_tilde[:, m], R_tilde[:, n]))
            else:
                d_list.append(0)
        value_list, indices_list = order_list(d_list)
        k_nei_value = value_list[:k]
        k_nei_index = indices_list[:k]  # 取前k个近似相关的value和index

        for i in range(len(R_tilde[:, m])):
            if np.isnan(R[i, m]) == True:  
                sum_tilde = 1 / sum(abs(x) for x in k_nei_value)
                sum_e = 0
                for o in range(k):
                    if np.isnan(R_tilde[i, k_nei_index[o]]) == False:
                        sum_e += k_nei_value[o] * R_tilde[i, k_nei_index[o]]

                R[i, m] = R_hat[i, m] + sum_tilde * sum_e

    for row in validation:
        R[row[1], row[0]] = row[2]

    return R


# print(main(mr.predictedRatings))
# np.savetxt("predictedRatings.txt", main(mr.predictedRatings))
