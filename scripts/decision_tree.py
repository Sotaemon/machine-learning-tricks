import numpy as np
import pandas as pd


class Node:
    def __init__(self):
        self.feat = None
        self.split = None
        self.child = []


class DecisionTree:
    def __init__(self, X, Y, feat_ranges, lbd):
        self.root = Node()
        self.X = X
        self.Y = Y
        self.feat_ranges = feat_ranges
        self.lbd = lbd
        self.eps = 1e-8
        self.T = 0
        self.ID3(self.root, self.X, self.Y)

    def aloga(self, a):
        return a * np.log2(a + self.eps)

    def entropy(self, Y):
        cnt = np.unique(Y, return_counts=True)[1]
        N = len(Y)
        ent = -np.sum([self.aloga(Ni / N) for Ni in cnt])
        return ent

    def info_gain(self, X, Y, feat, val):
        N = len(Y)
        if N == 0:
            return 0
        HX = self.entropy(Y)
        HXY = 0
        Y_l = Y[X[:, feat] <= val]
        HXY += len(Y_l) / len(Y) * self.entropy(Y_l)
        Y_r = Y[X[:, feat] > val]
        HXY += len(Y_r) / len(Y) * self.entropy(Y_r)
        return HX - HXY

    def entropy_YX(self, X, Y, feat, val):
        HYX = 0
        N = len(Y)
        if N == 0:
            return 0
        Y_l = Y[X[:, feat] <= val]
        HYX += -self.aloga(len(Y_l) / N)
        Y_r = Y[X[:, feat] > val]
        HYX += -self.aloga(len(Y_r) / N)
        return HYX

    def info_gain_ratio(self, X, Y, feat, val):
        IG = self.info_gain(X, Y, feat, val)
        HYX = self.entropy_YX(X, Y, feat, val)
        return IG / HYX

    def ID3(self, node, X, Y):
        if len(np.unique(Y)) == 1:
            node.feat = Y[0]
            self.T += 1
            return
        best_IGR = 0
        best_feat = None
        best_val = None
        for feat in range(len(feat_names)):
            for val in self.feat_ranges[feat_names[feat]]:
                IGR = self.info_gain_ratio(X, Y, feat, val)
                if IGR > best_IGR:
                    best_IGR = IGR
                    best_feat = feat
                    best_val = val
        cur_cost = len(Y) * self.entropy(Y) + self.lbd
        if best_feat is None:
            new_cost = np.inf
        else:
            new_cost = 0
            X_feat = X[:, best_feat]
            new_Y_l = Y[X_feat <= best_val]
            new_cost += len(new_Y_l) * self.entropy(new_Y_l)
            new_Y_r = Y[X_feat > best_val]
            new_cost += len(new_Y_r) * self.entropy(new_Y_r)
            new_cost += 2 * self.lbd

        if new_cost <= cur_cost:
            node.feat = best_feat
            node.split = best_val
            l_child = Node()
            l_X = X[X_feat <= best_val]
            l_Y = Y[X_feat <= best_val]
            self.ID3(l_child, l_X, l_Y)
            r_child = Node()
            r_X = X[X_feat > best_val]
            r_Y = Y[X_feat > best_val]
            self.ID3(r_child, r_X, r_Y)
            node.child = [l_child, r_child]
        else:
            vals, cnt = np.unique(Y, return_counts=True)
            node.feat = vals[np.argmax(cnt)]
            self.T += 1

    def predict(self, x):
        node = self.root
        while node.split is not None:
            if x[node.feat] <= node.split:
                node = node.child[0]
            else:
                node = node.child[1]
        return node.feat

    def accuracy(self, X, Y):
        correct = 0
        for x, y in zip(X, Y):
            pred = self.predict(x)
            if pred == y:
                correct += 1
        return correct / len(Y)


# 读取数据
data = pd.read_csv("./../data/titanic/train.csv")
print(data.info())
print(data[:5])
# 去除无用数据项
data.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
# 设置参数
feat_ranges = {}
cont_feat = ["Age", "Fare"]
bins = 10
# 获取数据范围
for feat in cont_feat:
    min_val = np.nanmin(data[feat])
    max_val = np.nanmax(data[feat])
    feat_ranges[feat] = np.linspace(min_val, max_val, bins).tolist()
    print(feat, ": ")
    for spt in feat_ranges[feat]:
        print(f"{spt:.4f}")
#
cat_feat = ["Sex", "Pclass", "SibSp", "Parch", "Cabin", "Embarked"]
for feat in cat_feat:
    data[feat] = data[feat].astype("category")
    print(f"{feat}：{data[feat].cat.categories}")
    data[feat] = data[feat].cat.codes.to_list()
    ranges = list(set(data[feat]))
    ranges.sort()
    feat_ranges[feat] = ranges
# 填充空缺数据为 -1
data.fillna(-1, inplace=True)
for feat in feat_ranges.keys():
    feat_ranges[feat] = [-1] + feat_ranges[feat]
#
np.random.seed(0)
feat_names = data.columns[1:]
label_name = data.columns[0]

data = data.reindex(np.random.permutation(data.index))
ratio = 0.8
split = int(ratio * len(data))
train_x = data[:split].drop(columns=["Survived"]).to_numpy()
train_y = data["Survived"][:split].to_numpy()
test_x = data[split:].drop(columns=["Survived"]).to_numpy()
test_y = data["Survived"][split:].to_numpy()
print("训练集大小：", len(train_x))
print("测试集大小：", len(test_x))
print("特征数：", train_x.shape[1])

DT = DecisionTree(train_x, train_y, feat_ranges, lbd=1.0)
print("叶结点数量：", DT.T)
print("训练集准确率：", DT.accuracy(train_x, train_y))
print("测试集准确率：", DT.accuracy(test_x, test_y))

from sklearn import tree

c45 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6)
c45.fit(train_x, train_y)

cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=6)
cart.fit(train_x, train_y)

c45_train_pred = c45.predict(train_x)
c45_test_pred = c45.predict(test_x)
cart_train_pred = cart.predict(train_x)
cart_test_pred = cart.predict(test_x)
print(
    f"训练集准确率：C4.5：{np.mean(c45_train_pred == train_y)}，"
    f"CART：{np.mean(cart_train_pred == train_y)}"
)
print(
    f"测试集准确率：C4.5：{np.mean(c45_test_pred == test_y)}，"
    f"CART：{np.mean(cart_test_pred == test_y)}"
)

import pydotplus
from six import StringIO

dot_data = StringIO()
tree.export_graphviz(
    c45,
    out_file=dot_data,
    feature_names=feat_names,
    class_names=["non-survival", "survival"],
    filled=True,
    rounded=True,
    impurity=False,
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png("./../results/tree.png")
graph.write_png("./../results/tree.png")
