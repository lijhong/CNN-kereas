''' 添加库'''
import numpy as np
import pandas as pd
''' 载入文件'''
header = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('mk-100k/u.data',sep='\t',names = header)

''' 读取数据'''
n_user = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_user) + '| Number of movies = ' + str(n_items)
## 显示用户的数量，电影的数量
''' 将数据集分为测试集与训练集 '''
from sklearn import cross_validation as cv
train_data,test_data = cv.train_test_split(df,test_size = 0.25)

''' 创建用户-产品矩阵,包含测试数据和训练数据'''
train_data_matrix = np.zero(n_user,n_items)
for line in train_data.itertuples():
    train_data_matrix[line[1]-1,lin[2]-1] = line[3]

test_data_matrix = np.zero((n_user,n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1] = line[3]


''' 利用sklearn 的 pairwise_distances 函数
计算余弦相似性
输出范围是0-1，因为打分都是正的'''

from sklearn.metrics.pairwise import pairwise_distances
user_similartity = pairwise_distances(train_data_matrix,metric='cosine')
item_similartity = pairwise_distances(train_data_matrix.T,metric='cosine')

''' 进行预测'''
def predict(ratings,similarity,type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        rating_diff = (ratings - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(ratings.diff)/np.array[np.abs((similarity),sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_predicition = predict(train_data_matrix,item_similartity,type='item')
user_predicition = predict(train_data_matrix,user_similartity,type='user')

''' 评估使用RMSE（均方根误差）'''

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction,ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))

''' 基于内存的CF显示误差值 '''
print 'User_based CF RMSE : ' + str(rmse(user_predicition,test_data_matrix))
print 'item_based CF RMSE : ' + str(rmse(item_predicition,test_data_matrix))


''' 基于模型的协同过滤 '''

''' 计算movielens数据集的稀疏度 '''
sparsity = round(1.0 -len(df)/float(n_user * n_items),3)
print 'The sparsity level of movielens100k is ' + str(sparsity*100) + '%'

import scipy.sparse as sp
from scipy.sparse.linalg import svds

''' 计算基于模型的过滤据方根误差'''
u,s,vt = svds(train_data_matrix,k = 20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
print 'User-based CF MSE:' + str(rmse(X_pred,test_data_matrix))

