import numpy as np
from numpy import array

#求解正规方程
'''
 y = X*w + b
 y = w1 * x1 + w2 * x2 + w3 * x3 + b
 y = w1 * x1 + w2 * x2 + w3 * x3 + w0
 y = w1 * x1 + w2 * x2 + w3 * x3 + w0*1
 w = (X^T * X)^-1 * X^T * y
w:系数
b:截距
1、( X^T ) 表示矩阵 ( X ) 的转置。
2、( (X^T * X)^{-1} ) 表示矩阵 ( X^T * X ) 的逆矩阵。
3、( X^T * y ) 表示矩阵 ( X ) 的转置与目标向量 ( y ) 的乘积。 最终结果 ( w ) 是通过上述矩阵运算得到的回归系数
eg:
x + y = 14
2x - y = 10
'''
# x = array([[1, 1], [2, -1]])
# y = array([14, 10])
# w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# print(w)

'''
三元一次方程
x-y+z = 100
2x+y-z=80
3x-2y+6z=256
'''
# x = array([[1, -1, 1], [2, 1, -1], [3, -2, 6]])
# y = array([100, 80, 256])
# w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# print(w)
'''
八元一次方程组

'''
x = array([[0,14,8,0,5,-2,9,-3],
           [-4,10,6,4,-14,-2,-14,8],
           [-1,-6,5,-12,3,-3,2,-2],
           [5,-2,3,10,5,11,4,-8],
           [-15,-15,-8,-15,7,-4,-12,2],
           [11,-10,-2,4,3,-9,-6,7],
           [-14,0,4,-3,5,10,13,7],
           [-3,-7,-2,-8,0,-6,-5,-9]])
y = array([339,-114,30,126,-395,-87,422,-309])
z = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# print(z)

# sklearn 库的使用
from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=False)
# model.fit(x,y)
#coef_是值,coef是系数含义
#filt_intercept_是截距，如果是false 那么model.intercept_就是0
# b_= model.intercept_
# w_ = model.coef_
# print(b_)
# print(w_)
#计算预测值
# w = x.dot(w_) + b_
# print(w)

#修改x值，以前的方程是没有加b的 现在我们需要加上b 比如b的值是12
# x= np.concatenate([x,np.full((8,1),1)],axis=1)
# model.fit(x,y)
# print(model.coef_)
# print(model.intercept_)
# print(x.dot(model.coef_) + model.intercept_)

'''
作业跟决高度计算温度
求解w,b
0*w +b = 12.834044
500* w  +b = 10.190649
1000* w +b = 5.500229
1500* w +b = 2.854665
2000* w +b = -0.706488
2500* w +b = -4.065323
3000* w +b = -7.127480
3500* w +b = -10.058879
4500* w +b = -13.206465
'''
he = array([[0, 1],
            [500, 1],
            [1000, 1],
            [1500, 1],
            [2000, 1],
            [2500, 1],
            [3000, 1],
            [3500, 1],
            [4500, 1]])
te = array([12.834044, 10.190649, 5.500229, 2.854665, -0.706488, -4.065323, -7.127480, -10.058879, -13.206465])
# model = LinearRegression(fit_intercept=True)
# model.fit(he, te)
# w1 = model.coef_
# b1 = model.intercept_
# # print(he.dot(w1) + b1)
# print(f"正规解:w0 = {w1[0]:.8f},w1 = {w1[1]:.8f}")
# print(model.intercept_)
# y1= np.linalg.inv(he.T.dot(he)).dot(he.T).dot(te)
# print(y1)
#
#
# 正规方程解法
XTX = he.T @ he
XTy = he.T @ te
#inv 求逆矩阵 就是代表公式的-1
w_b = np.linalg.inv(XTX) @ XTy
w = w_b[0]

# 最小二乘法解法 (推荐)
# w_lstsq, b_lstsq = np.linalg.lstsq(he, te, rcond=None)[0]
#
# # 计算预测值和误差
# y_pred = he @ np.array([w, b])
# residuals = te - y_pred
# mse = np.mean(residuals**2)

#打印结果
print(f"正规方程结果: w = {w:.8f}")
# print(f"最小二乘法结果: w = {w_lstsq:.8f}, b = {b_lstsq:.8f}")
# print(f"均方误差 (MSE): {mse:.8f}")
# #计算8000米的温度
# he8 = array([[8000,1]])
# te8 = he8.dot(w1) + b1
# print(f"8000米的温度是：,{te8[0]:.8f}")

