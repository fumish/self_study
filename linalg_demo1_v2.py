# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 技術者のための線形代数学 1章 デモ
# 1. 1次変換
#     + 行列の要素を色々変更するとどうなるか
#
# 2. 固有値関連の内容
#
# 3. 最小二乗法

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

# # 1次変換
# + 教科書から1次変換は
# $$
# \left[\begin{array}{c}
#     x' \\
#     y' \\
# \end{array}\right] = 
# A
# \left[\begin{array}{c}
#     x \\
#     y \\
# \end{array}\right]
# $$
# ここでは、画像の拡大、せん断、回転を見てみる

X = np.array([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]])
(pos_x, pos_y) = np.where(X == 1)

# ## 対象となる画像:

plt.imshow(X)
plt.show()


# ## 画像の拡大
# + 拡大行列$P(\alpha, \beta)$は
# $$
# P(\alpha, \beta) = \left[\begin{array}{cc}
#     \alpha & 0 \\
#     0 & \beta \\
# \end{array}\right]
# $$
# 特に$(a,b)$周りで拡大したとき、変換後の位置$(x', y')$は
# $$
# \left[\begin{array}{c}
#     x' \\
#     y' \\
# \end{array}\right] = 
# \left[\begin{array}{cc}
#     \alpha & 0 \\
#     0 & \beta \\
# \end{array}\right]
# \left[\begin{array}{c}
#     x - a \\
#     y - b \\
# \end{array}\right]
# +
# \left[\begin{array}{c}
#     a \\
#     b \\
# \end{array}\right]
# $$
# + ** 画像の行き先をドット上にしてするため四捨五入しているので割と粗いです **

def im_extend(alpha, beta, offset_x, offset_y):
    """
    画像を拡大する関数
    """
    P = np.array([[alpha, 0], [0, beta]])
    pos_dash = np.zeros((len(pos_x), 2))
    X_dash = np.zeros(X.shape)
    for i, (i_pos_x, i_pos_y) in enumerate(zip(pos_x, pos_y)):
        pos_dash[i,:] = np.round(P @ np.array([i_pos_x - offset_x, i_pos_y - offset_y])) + np.array([offset_x, offset_y])
        
    for i in range(len(pos_x)):
        X_dash[int(pos_dash[i,0]), int(pos_dash[i,1])] = X[pos_x[i], pos_y[i]]
    print("元の画像:")
    plt.imshow(X)
    plt.show()
    
    print("1次変換の行列:")
    print(P)
    print("1次変換適用後の画像:")
    plt.imshow(X_dash)
    plt.show()


interact(im_extend, alpha = (0, 2, 0.1), beta=(0, 2, 0.1), offset_x = (0, 7, 1), offset_y=(0, 7, 1))


# ## 画像の拡大
# + 拡大行列$P(\alpha, \beta)$は
# $$
# Q(\gamma) = \left[\begin{array}{cc}
#     1 & \gamma \\
#     0 & 1 \\
# \end{array}\right]
# $$
# 特に$(a,b)$周りで拡大したとき、変換後の位置$(x', y')$は
# $$
# \left[\begin{array}{c}
#     x' \\
#     y' \\
# \end{array}\right] = 
# \left[\begin{array}{cc}
#     1 & \gamma \\
#     0 & 1 \\
# \end{array}\right]
# \left[\begin{array}{c}
#     x - a \\
#     y - b \\
# \end{array}\right]
# +
# \left[\begin{array}{c}
#     a \\
#     b \\
# \end{array}\right]
# $$
# + ** 画像の行き先をドット上にしてするため四捨五入しているので割と粗いです **

def im_shear(gamma, offset_x, offset_y):
    """
    画像をせん断する関数
    """
    Q = np.array([[1, gamma], [0, 1]])
    pos_dash = np.zeros((len(pos_x), 2))
    X_dash = np.zeros(X.shape)
    for i, (i_pos_x, i_pos_y) in enumerate(zip(pos_x, pos_y)):
        pos_dash[i,:] = np.round(Q @ np.array([i_pos_x - offset_x, i_pos_y - offset_y])) + np.array([offset_x, offset_y])
        
    for i in range(len(pos_x)):
        X_dash[int(pos_dash[i,0]), int(pos_dash[i,1])] = X[pos_x[i], pos_y[i]]
    print("元の画像:")
    plt.imshow(X)
    plt.show()
    
    print("1次変換の行列:")
    print(Q)
    print("1次変換適用後の画像:")
    plt.imshow(X_dash)
    plt.show()


interact(im_shear, gamma=(-2.7, 2.7, 0.1), offset_x = (0, 7, 1), offset_y=(0, 7, 1))


# ## 画像を回転させる
# + 回転行列$R(\theta)$は
# $$
# R(\theta) = \left[\begin{array}{cc}
#     \cos(\theta) & -\sin(\theta) \\
#     \sin(\theta) & \cos(\theta) \\
# \end{array}\right]
# $$
# 特に$(a,b)$周りで回転したとき、回転先の位置$(x', y')$は
# $$
# \left[\begin{array}{c}
#     x' \\
#     y' \\
# \end{array}\right] = 
# \left[\begin{array}{cc}
#     \cos(\theta) & -\sin(\theta) \\
#     \sin(\theta) & \cos(\theta) \\
# \end{array}\right]
# \left[\begin{array}{c}
#     x - a \\
#     y - b \\
# \end{array}\right]
# +
# \left[\begin{array}{c}
#     a \\
#     b \\
# \end{array}\right]
# $$
# + ** 画像の行き先をドット上にしてするため四捨五入しているので割と粗いです **

def im_rotate(theta, offset_x, offset_y):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pos_dash = np.zeros((len(pos_x), 2))
    X_dash = np.zeros(X.shape)
    for i, (i_pos_x, i_pos_y) in enumerate(zip(pos_x, pos_y)):
        pos_dash[i,:] = np.round(R @ np.array([i_pos_x - offset_x, i_pos_y - offset_y])) + np.array([offset_x, offset_y])
        
    for i in range(len(pos_x)):
        X_dash[int(pos_dash[i,0]), int(pos_dash[i,1])] = X[pos_x[i], pos_y[i]]
    print("元の画像:")
    plt.imshow(X)
    plt.show()

    print("1次変換の行列:")
    print(R)
    print("1次変換適用後の画像:")
    plt.imshow(X_dash)
    plt.show()


interact(im_rotate, theta=(-np.pi, np.pi+0.1, 0.1*np.pi), offset_x = (0, 7, 1), offset_y=(0, 7, 1))

# # 固有値関連の内容
# + データの次元を圧縮する

n = 200
data_seed = 20191012
np.random.seed(data_seed)

(a,b) = (-3, 5)

# ## $x \sim U(-10, 10)$, $y=ax+b + noise$に従ってデータを発生させる

domain_X = (-5, 5)
X = np.random.uniform(low = domain_X[0], high = domain_X[1], size = n)
Y = a*X + b + np.random.normal(scale = 4, size = n)
original_data = np.array([X,Y]).T

test_X = np.linspace(start = domain_X[0], stop = domain_X[1], num = 100)
test_Y = a*test_X + b

plt.scatter(X,Y)
plt.show()

# ## 情報は$ax + b$に集約されているので、この(x,y)を線分上に射影したい

chose_ind = np.random.permutation(n)
chose_num = 5
plt.plot(test_X, test_Y)
for i in range(chose_num):
    d = np.array([1, a])
    p0 = np.array([0, b])
    p = (X[chose_ind[i]], Y[chose_ind[i]])
    length = (-d@(p0-p))/(d@d)
    OS = p0 + length*d
    data_X = np.array([p[0], OS[0]])
    data_Y = np.array([p[1], OS[1]])
    plt.scatter(data_X, data_Y)
plt.show()

# ## 求め方
# + 以下の関数$L(w)$の最大化を行えばよい
# $$
# L(w) = \sum_{i=1}^n (x_i^T w)^2
# $$
# 感覚的には、x_iのばらつきが大きい方向を見つけて、そのばらつきを除去するようなことをしている。  
# $w^T w = 1$の制約の下でこれを解くと、$X^T X$に対する教科書の固有値問題を解くのと同じになる:

norm_mean = original_data.mean(axis = 0)
norm_data = original_data - norm_mean

cov_mat = norm_data.T @ norm_data


def is_pos_def(A:np.ndarray):
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return False
    return True


### 正定値性のチェック
if is_pos_def(cov_mat):
    [eig_val, eig_vec] = np.linalg.eigh(cov_mat)
    compressed_proj = eig_vec[:,1]

proj_data = (norm_data @ compressed_proj).reshape(n,1) @ compressed_proj.reshape(1,2) + norm_mean

# +
# focus_ind = 9
# plt.plot(test_X, test_Y)
# plt.scatter(original_data[focus_ind,0], original_data[focus_ind,1])
# plt.scatter(proj_data[focus_ind,0], proj_data[focus_ind,1])
# plt.show()
# -

plt.plot(test_X, test_Y)
# plt.scatter(original_data[:,0], original_data[:,1])
plt.scatter(proj_data[:,0], proj_data[:,1])
plt.show()

# # 最小二乗法

# ## 普通の場合
# + $L(w) = \sum_{i=1}^n (y_i - x_i^Tw)^T (y_i - x_i^Tw)$, where $x_i, w \in \mathbb{R}^2$について  
# + $L(w)$を最小にする$w$($(=\hat{w}$とする)を見つける問題を考えたい
#     + 点が与えられて、点をいい感じに通るような線が知りたい
#         + 応用: 点: (年齢, 平均寿命)でデータがない部分の平均寿命が知りたい etc.

n = 20
data_seed = 20191012
np.random.seed(data_seed)

domain_X = (-5, 5)
train_X = np.ones((n,2))
train_X[:,1] = np.random.uniform(low = domain_X[0], high = domain_X[1], size = n)
true_w = np.array([-2, 3])
train_Y = train_X @ true_w + np.random.normal(scale = 1, size = n)

plt.scatter(train_X[:,1], train_Y)
plt.show()

# ## 求め方
# + $L(w)$は行列で書くとシンプルに書ける(白板で示します):
#     + $L(w) = (y - Xw)^T (y - Xw)$
# + $L(w)$は2次関数の一般化みたいなものなので、2次関数の最小問題のようにすれば解ける:
#     + $L(w) = (w - \hat{w})^T A (w - \hat{w})$みたいにかけるはずで、$\hat{w},A$を平方完成(数Iを参照)の一般化をすればよい













# + 答え
# $\hat{w} = (X^T X)^{-1}X y$

est_mle_w = np.linalg.solve(train_X.T @ train_X, train_X.T @ train_Y)

test_X_range = np.linspace(start = -5, stop = 5, num = 20)
test_X = np.ones((len(test_X_range),2))
test_X[:,1] = test_X_range
test_true_Y = test_X @ true_w
test_est_mle_Y = test_X @ est_mle_w

plt.plot(test_X[:,1], test_true_Y, label ="true")
plt.plot(test_X[:,1], test_est_mle_Y, label = "mle")
plt.legend()
plt.show()

# ## 過学習の場合
# + $x_i$の値に引っ張られたり、$(X^T X)^{-1}$は固有値に0を含むとうまく計算できない
#     + $det(X^T X) = \lambda_1 * \lambda_2$で、行列式が0になるため
# + $L(w) = \sum_{i=1}^n (y_i - x_i^Tw)^T (y_i - x_i^Tw)$, where $x_i, w \in \mathbb{R}^2 + \beta w^T w$を考える($\beta > 0$)と固有値0問題は実は解決する:
#     + $(X^T X)^{-1} \rightarrow (X^T X + \beta I)^{-1}$
# + $A + B$の固有値は$\lambda_A + \lambda_B$で($\lambda_A$は$A$の固有値, $\lambda_B$は$B$の固有値), $\beta I$の固有値は$\beta$, $X^T X$の固有値は実は$\geq 0$ (証明は今後出てくるかもしれない)なので、$X^T X + \beta I$の固有値は$>0$
#

n = 10
data_seed = 20191012
np.random.seed(data_seed)

domain_X = (-5, 5)
train_X = np.ones((n,2))
train_X[:,1] = np.random.uniform(low = domain_X[0], high = domain_X[1], size = n)
true_w = np.array([2, 3])
train_Y = train_X @ true_w + np.random.normal(scale = 4, size = n)

np.linalg.eigh(train_X.T @ train_X)

beta = 2

# +
# train_Y[0] = 10
# train_Y[2] = 15
# -

plt.scatter(train_X[:,1], train_Y)
plt.show()

est_ridge_w = np.linalg.solve(train_X.T @ train_X + beta * np.eye(2), train_X.T @ train_Y)
est_mle_w = np.linalg.solve(train_X.T @ train_X, train_X.T @ train_Y)

test_X_range = np.linspace(start = -5, stop = 5, num = 20)
test_X = np.ones((len(test_X_range),2))
test_X[:,1] = test_X_range
test_true_Y = test_X @ true_w
test_est_mle_Y = test_X @ est_mle_w
test_est_ridge_Y = test_X @ est_ridge_w

est_mle_w, est_ridge_w

# +
# np.linalg.solve(train_X.T @ train_X + beta * np.eye(2), train_X.T)

# +
# np.linalg.solve(train_X.T @ train_X, train_X.T)
# -

plt.plot(test_X[:,1], test_true_Y, label = "true")
plt.plot(test_X[:,1], test_est_mle_Y, label ="mle")
plt.plot(test_X[:,1], test_est_ridge_Y, label = "ridge")
plt.legend()
plt.show()

# + mleが引っ張られる理由はラグランジュの未定乗数法から出るが、線形代数学の範囲外なので割愛
# + 


