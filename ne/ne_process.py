import scipy.io as sio
import numpy as np
import numpy.matlib as npmat
import scipy
from numpy import linalg as LA
import math
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


def save_hotmap(network, address):
    plt.imshow(network)
    image = address.split('.')[0] + '.png'
    image_url = os.getcwd().replace('\\', '/') + '/static/images/' + image
    plt.savefig(image_url)
    return image


def network_ne(fname, data, alpha=0.9):
    for item in data.keys():
        try:
            if data[item].ndim == 2:
                if len(data[item]) == len(data[item][0]):
                    raw = data[item]
                    matrix_key = item
        except Exception:
            print(Exception)

    save_hotmap(raw, fname)
    # 原始网络大小

    raw_size = raw.shape[0]

    # 对角线用0替代
    # np.fill_diagonal(raw, 0)
    # print(raw)
    # m_nonzero_rows = raw[[i for i, x in enumerate(raw) if x.any()]]
    # 稀疏表示
    raw = scipy.sparse.csr_matrix(raw)
    # 统计每一行非零元素个数
    num_nonzeros = np.diff(raw.indptr)
    # 记录零元素所在位置
    index_zeros = [i for i in range(len(num_nonzeros)) if num_nonzeros[i] == 0]
    # 构造一个向量，非零元素位置为1，零元素位置为0
    nonzeros = np.ones(raw.shape[0])
    for i in index_zeros:
        nonzeros[i] = 0

    # 将全为0的行和列从网络中删除
    raw = raw[raw.getnnz(1) > 0][:, raw.getnnz(0) > 0].toarray()

    # load network and information
    shape = raw.shape
    node_num = raw.shape[0]

    magnify = np.diag(raw.sum(axis=0))

    # calculate matrix p
    # temp = np.multiply(raw, knn)
    # p = np.divide(temp, temp.sum(axis=1).reshape(-1, 1))  # array saving transition probability matrix P

    p = np.divide(raw * node_num, (raw * node_num).sum(axis=1).reshape(-1, 1))

    print(p)

    p = (p + p.T) / 2

    # construct knn
    K = 20
    if (node_num / 10) < 20:
        K = math.ceil(node_num / 10)

    knn = np.zeros(shape, dtype=np.int)  # array saving k nearest neighbors of every node
    # construct knn
    i = 0
    for line in p:
        knn_index = np.argpartition(line, -K)[-K:]
        for index in knn_index:
            knn[i][index] = 1
        i = i + 1

    p = np.multiply(p, knn)

    p = (p + p.T) / 2

    p = p + (np.eye(node_num) + np.diag(p.sum(axis=1)))

    print(p)
    # magnify = np.diag(temp.sum(axis=1))
    # print("概率转移矩阵")
    # print(p)

    # calculate matrix t
    # t_denominators = p.sum(axis=0).reshape(-1, 1)

    # t = np.dot(p, np.divide(p.T, t_denominators))  # array saving local network T

    t = np.divide(p * node_num, ((p * node_num).sum(axis=1) + np.finfo(np.float64).eps).reshape(-1, 1))

    wr = np.sqrt(t.sum(axis=0))

    t = np.divide(t, np.matlib.repmat(wr, node_num, 1))

    t = np.dot(t, t.T)

    print(t)

    # print("局部结构")
    # print(t)
    '''
    wwtf = raw
    for i in range(50):
        wwtf = np.dot(np.dot(t, wwtf), t) * alpha + (1 - alpha) * t

    print("迭代结果")
    print(wwtf)
    '''
    # 收敛结果
    # w = (1 - alpha) * np.dot(t, (npmat.eye(node_num) - alpha * np.dot(t, t)).I)

    eigValue, v= LA.eig(t)


    l = np.argsort(eigValue)
    eigValue=np.sort(eigValue)
    eigVector = v[:, l]
    print(eigValue)
    print(eigVector)

    d = np.divide((1 - alpha) * eigValue, 1 - alpha * eigValue * eigValue)
    print(d)

    D = np.diag(d)
    print(D)
    w = np.dot(np.dot(eigVector, D), eigVector.T)
    print(w.shape)
    w = w * (1 - np.eye(node_num)) / npmat.repmat(1-np.diag(w),node_num,1)

    np.fill_diagonal(w, 0)

    print(w)
    # r=[[w[i][j]for j in range(len(nonzeros)) if nonzeros[j]==1]for i in range(len(nonzeros)) if nonzeros[i]==1]

    w = np.dot(magnify, w)

    w = (w + w.T) / 2

    print(w)

    zero_row = np.zeros(w.shape[0])
    zero_line = np.zeros(raw_size)

    for i in range(len(nonzeros)):
        if nonzeros[i] == 0:
            w = np.insert(w, i, values=zero_row, axis=0)

    for i in range(len(nonzeros)):
        if nonzeros[i] == 0:
            w = np.insert(w, i, values=zero_line, axis=1)

    # print("收敛公式结果")
    # print(w.shape)
    # print(w)

    image = save_hotmap(w, 'ne_' + fname)

    os.remove(fname)

    ext = fname.split('.')[1]

    if ext == 'mat':
        ne_path = 'ne_' + fname.split('.')[0] + '.mat'
        data[matrix_key] = w
        sio.savemat(ne_path, data)
    elif ext == 'txt':
        ne_path = 'ne_' + fname.split('.')[0] + '.txt'

        np.savetxt(ne_path, w, delimiter=',',fmt='%.4f')
    elif ext=='csv':
        ne_path = 'ne_' + fname.split('.')[0] + '.csv'

        df=pd.DataFrame(w)
        df.to_csv(path_or_buf = ne_path,header=data['label'],float_format='%.4f',index=False)


      #  np.savetxt(ne_path, w, delimiter=',', fmt='%.4f', header=data['label'])


    return ne_path
