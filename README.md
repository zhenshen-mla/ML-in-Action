# ML-in-Action
## Perceptron
感知机是典型的分类模型。通过迭代训练来更新参数值，对权重w的更新为 ```lr*(target_i-predict(x_i))*x_i```，对偏执bias的更新为```lr*(target_i-predict(x_i))```。
## Logistic Regression
感知机与逻辑回归的异同：  
同  
（1）两者都为线性分类器，只能处理线性可分的数据。  
（2）两者的优化方法可以统一为GD\SGD。GD是每个训练样本都会出发参数更新，SGD是整个训练迭代结束后进行参数更新。  
异  
（1）两者的损失函数有所不同，PLA针对误分类点到超平面的距离总和进行建模，LR使用交叉熵损失建模。  
LR比PLA的优点之一在于对于激活函数的改进。
前者为sigmoid function，后者为step function。
LR使得最终结果有了概率解释的能力（将结果限制在0-1之间），sigmoid为平滑函数（连续可导），能够得到更好的分类结果，
而step function为分段函数，对于分类的结果处理比较粗糙，非0即1，而不是返回一个分类的概率。
## Linear Regression
(1)
回归模型正是表示从输入变量到输出变量之间映射的函数。
回归问题的学习等价于函数拟合：选择一条函数曲线使其很好地拟合已知数据且很好地预测未知数据。
回归就是要尽可能的把所有的样本点拟合到一条曲线上，尽管有的样本点不满足曲线，但是要使得离曲线距离尽可能近。
(2)
分为两个类别：简单线性回归和岭回归（岭回归是添加L2正则化约束）
(3)
实现问题：在公式推导中，我们已经将W矩阵利用样本的特征与label进行了表示，故直接运算即可，不需要进行迭代
(4)
为什么引入岭回归：防止过拟合；这种情况是数据的特征比样本还多，此时的矩阵X不是满秩矩阵，无法求逆。岭回归就是在普通线性回归的基础上引入单位矩阵，使其可逆

## SVM
SVM有三宝，间隔、对偶、核技巧  
定义：在特征空间上的间隔最大的线性分类器，即求解能够正确划分训练数据集并且几何间隔最大的分离超平面。  
区别于感知机：对于线性可分的数据集来说，感知机划分的超平面有无穷多个，但是几何间隔最大的分离超平面却是唯一的
  
## 推导  
 <div align=left><img width="1000" height="1700" src="https://github.com/zhenshen-mla/Support-Vector-Machine/blob/master/examples/total.png"/></div>  
 
## Linear Regression

回归用于预测输入变量和输出变量之间的关系，特别是当输入变量的值发生变化时，输出变量的值随之发生变化。回归模型正是表示从输入变量到输出变量之间映射的函数。
回归问题的学习等价于函数拟合：选择一条函数曲线使其很好地拟合已知数据且很好地预测未知数据。
回归就是要尽可能的把所有的样本点拟合到一条曲线上，尽管有的样本点不满足曲线，但是要使得离曲线距离尽可能近。  

  
## 推导  
 <div align=left><img width="1000" height="1500" src="https://github.com/zhenshen-mla/Support-Vector-Machine/blob/master/examples/linear regression.jpg"/></div>  
