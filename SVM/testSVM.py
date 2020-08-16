import SVM.smo as sm

# 通过训练数据计算 b, alphas
dataArr, labelArr = sm.loadDataSet('F:\Machine Learning\SVM\TrainData.txt')
# print(dataArr) [[-0.214824, 0.662756], [-0.061569, -0.091875], [0.406933, 0.648055],.... ]
# print(labelArr)  [-1.0, 1.0, -1.0, 1.0, -1.0,...]

b, alphas = sm.smoP(dataArr, labelArr, C=200, toler=0.0001, maxInter=10000, kTup=('rbf', 0.10))

sm.drawDataMap(dataArr, labelArr, b, alphas)
sm.getTrainingDataResult(dataArr, labelArr, b, alphas, 0.10)
dataArr1, labelArr1 = sm.loadDataSet('F:\Machine Learning\SVM\TestData.txt')
# 测试结果
sm.getTestDataResult(dataArr1, labelArr1, b, alphas, 0.10)

