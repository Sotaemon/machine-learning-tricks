import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import make_classification, make_friedman1
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

class RandomForest():
    def __init__(self, n_trees=10, max_features='sqrt'):
        self.n_trees = n_trees
        self.oob_score = 0
        self.trees = [DTC(max_features=max_features)
                      for _ in range(n_trees)]
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.n_classes = np.unique(y).shape[0]
        ensemble = np.zeros((n_samples, self.n_classes))
        for tree in self.trees:
            idx = np.random.randint(0, n_samples, n_samples)
            unsampled_mask = np.bincount(idx, minlength=n_samples) == 0
            unsampled_idx = np.arange(n_samples)[unsampled_mask]
            tree.fit(x[idx], y[idx])
            ensemble[unsampled_idx] += tree.predict_proba(x[unsampled_idx])
        self.oob_score = np.mean(y == np.argmax(ensemble, axis=1))
    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    def predict_proba(self, x):
        ensemble = np.mean([tree.predict_proba(x)
                            for tree in self.trees], axis=0)
        return ensemble
    def score(self, x, y):
        return np.mean(y == self.predict(x))


class StackingClassifier():
    def __init__(
            self,
            classifiers,  # 基础分类器
            meta_classifier,  # 元分类器
            concat_feature=False,  # 是否将原始数据拼接在新数据上
            kfold=5  # K折交叉验证
    ):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.concat_feature = concat_feature
        self.kf = KFold(n_splits=kfold)
        # 为了在测试时计算平均，我们需要保留每个分类器
        self.k_fold_classifiers = []
    def fit(self, X, y):
        # 用X和y训练基础分类器和元分类器
        n_samples, n_features = X.shape
        self.n_classes = np.unique(y).shape[0]
        if self.concat_feature:
            features = X
        else:
            features = np.zeros((n_samples, 0))
        for classifier in self.classifiers:
            self.k_fold_classifiers.append([])
            # 训练每个基础分类器
            predict_proba = np.zeros((n_samples, self.n_classes))
            for train_idx, test_idx in self.kf.split(X):
                # 交叉验证
                clf = clone(classifier)
                clf.fit(X[train_idx], y[train_idx])
                predict_proba[test_idx] = clf.predict_proba(X[test_idx])
                self.k_fold_classifiers[-1].append(clf)
            features = np.concatenate([features, predict_proba], axis=-1)
        # 训练元分类器
        self.meta_classifier.fit(features, y)
    def _get_features(self, X):
        # 计算输入X的特征
        if self.concat_feature:
            features = X
        else:
            features = np.zeros((X.shape[0], 0))
        for k_classifiers in self.k_fold_classifiers:
            k_feat = np.mean([clf.predict_proba(X)
                              for clf in k_classifiers], axis=0)
            features = np.concatenate([features, k_feat], axis=-1)
        return features
    def predict(self, X):
        return self.meta_classifier.predict(self._get_features(X))
    def score(self, X, y):
        return self.meta_classifier.score(self._get_features(X), y)

x, y = make_classification(
    n_samples=1000,
    n_features=16,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    flip_y=0.1,
    random_state=0
)
print(x.shape)

num_trees = np.arange(1, 101, 5)
np.random.seed(0)
plt.figure()

oob_score = []
train_score = []
with tqdm(num_trees) as pbar:
    for n_tree in pbar:
        rf = RandomForest(n_trees=n_tree, max_features=None)
        rf.fit(x, y)
        train_score.append(rf.score(x, y))
        oob_score.append(rf.oob_score)
        pbar.set_postfix({
            'n_tree': n_tree,
            'train_score': train_score[-1],
            'oob_score': oob_score[-1]
        })
plt.plot(num_trees, train_score, color='blue',
    label='bagging_train_score')
plt.plot(num_trees, oob_score, color='blue', linestyle='-.',
    label='bagging_oob_score')

oob_score = []
train_score = []
with tqdm(num_trees) as pbar:
    for n_tree in pbar:
        rf = RandomForest(n_trees=n_tree, max_features='sqrt')
        rf.fit(x, y)
        train_score.append(rf.score(x, y))
        oob_score.append(rf.oob_score)
        pbar.set_postfix({
            'n_tree': n_tree,
            'train_score': train_score[-1],
            'oob_score': oob_score[-1]
        })
plt.plot(num_trees, train_score, color='red', linestyle='--',
    label='random_forest_train_score')
plt.plot(num_trees, oob_score, color='red', linestyle=':',
    label='random_forest_oob_score')

plt.ylabel('Score')
plt.xlabel('Number of trees')
plt.legend()
plt.show()

bc = BaggingClassifier(n_estimators=100, oob_score=True, random_state=0)
bc.fit(x, y)
print('bagging：', bc.oob_score_)

rfc = RFC(n_estimators=100, max_features='sqrt',
    oob_score=True, random_state=0)
rfc.fit(x, y)
print('随机森林：', rfc.oob_score_)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 基础分类器
rf = RFC(n_estimators=10, max_features='sqrt',
    random_state=0).fit(X_train, y_train)
knc = KNC().fit(X_train, y_train)
# multi_class='ovr'表示二分类问题
lr = LR(solver='liblinear', multi_class='ovr',
    random_state=0).fit(X_train, y_train)
print('随机森林：', rf.score(X_test, y_test))
print('KNN：', knc.score(X_test, y_test))
print('逻辑斯谛回归：', lr.score(X_test, y_test))
# 元分类器
meta_lr = LR(solver='liblinear', multi_class='ovr', random_state=0)

sc = StackingClassifier([rf, knc, lr], meta_lr, concat_feature=False)
sc.fit(X_train, y_train)
print('Stacking分类器：', sc.score(X_test, y_test))

# 带原始特征的stacking分类器
sc_concat = StackingClassifier([rf, knc, lr], meta_lr, concat_feature=True)
sc_concat.fit(X_train, y_train)
print('带原始特征的Stacking分类器：', sc_concat.score(X_test, y_test))

# 初始化stump
stump = DTC(max_depth=1, min_samples_leaf=1, random_state=0)

# 弱分类器个数
M = np.arange(1, 101, 5)
bg_score = []
rf_score = []
dsc_ada_score = []
real_ada_score = []
plt.figure()

with (tqdm(M) as pbar):
    for m in pbar:
        # bagging算法
        bc = BaggingClassifier(estimator=stump,
            n_estimators=m, random_state=0)
        bc.fit(X_train, y_train)
        bg_score.append(bc.score(X_test, y_test))
        # 随机森林算法
        rfc = RFC(n_estimators=m, max_depth=1,
            min_samples_leaf=1, random_state=0)
        rfc.fit(X_train, y_train)
        rf_score.append(rfc.score(X_test, y_test))
        # 离散 AdaBoost，SAMME是分步加性模型（stepwise additive model）的缩写
        dsc_adaboost = AdaBoostClassifier(estimator=stump,
            n_estimators=m, algorithm='SAMME', random_state=0)
        dsc_adaboost.fit(X_train, y_train)
        dsc_ada_score.append(dsc_adaboost.score(X_test, y_test))
        # 实 AdaBoost，SAMME.R表示弱分类器输出实数
        real_adaboost = AdaBoostClassifier(estimator=stump,
            n_estimators=m, algorithm='SAMME.R', random_state=0)
        real_adaboost.fit(X_train, y_train)
        real_ada_score.append(real_adaboost.score(X_test, y_test))

# 绘图
plt.plot(M, bg_score, color='blue', label='Bagging')
plt.plot(M, rf_score, color='red', ls='--', label='Random Forest')
plt.plot(M, dsc_ada_score, color='green', ls='-.', label='Discrete AdaBoost')
plt.plot(M, real_ada_score, color='purple', ls=':', label='Real AdaBoost')
plt.xlabel('Number of trees')
plt.ylabel('Test score')
plt.legend()
plt.tight_layout()
plt.savefig('output_26_1.png')
plt.savefig('output_26_1.pdf')
plt.show()

reg_X, reg_y = make_friedman1(
    n_samples=2000, # 样本数目
    n_features=100, # 特征数目
    noise=0.5, # 噪声的标准差
    random_state=0 # 随机种子
)

reg_X_train, reg_X_test, reg_y_train, reg_y_test = \
    train_test_split(reg_X, reg_y, test_size=0.2, random_state=0)

def rmse(regressor):
    y_pred = regressor.predict(reg_X_test)
    return np.sqrt(np.mean((y_pred - reg_y_test) ** 2))

xgbr = xgb.XGBRegressor(
    n_estimators=100, # 弱分类器数目
    max_depth=1, # 决策树最大深度
    learning_rate=0.5, # 学习率
    gamma=0.0, # 对决策树叶结点数目的惩罚系数，当弱分类器为stump时不起作用
    reg_lambda=0.1, # L2正则化系数
    subsample=0.5, # 与随机森林类似，表示采样特征的比例
    objective='reg:squarederror', # MSE损失函数
    eval_metric='rmse', # 用RMSE作为评价指标
    random_state=0 # 随机种子
)

xgbr.fit(reg_X_train, reg_y_train)
print(f'XGBoost：{rmse(xgbr):.3f}')

knnr = KNeighborsRegressor(n_neighbors=5).fit(reg_X_train, reg_y_train)
print(f'KNN：{rmse(knnr):.3f}')

lnr = LinearRegression().fit(reg_X_train, reg_y_train)
print(f'线性回归：{rmse(lnr):.3f}')

stump_reg = DecisionTreeRegressor(max_depth=1,
    min_samples_leaf=1, random_state=0)
bcr = BaggingRegressor(estimator=stump_reg,
    n_estimators=100, random_state=0)
bcr.fit(reg_X_train, reg_y_train)
print(f'Bagging：{rmse(bcr):.3f}')

rfr = RandomForestRegressor(n_estimators=100, max_depth=1,
    max_features='sqrt', random_state=0)
rfr.fit(reg_X_train, reg_y_train)
print(f'随机森林：{rmse(rfr):.3f}')

stkr = StackingRegressor(estimators=[
    ('knn', knnr),
    ('ln', lnr),
    ('rf', rfr)
])
stkr.fit(reg_X_train, reg_y_train)
print(f'Stacking：{rmse(stkr):.3f}')

stkr_pt = StackingRegressor(estimators=[
    ('knn', knnr),
    ('ln', lnr),
    ('rf', rfr)
], passthrough=True)
stkr_pt.fit(reg_X_train, reg_y_train)
print(f'带输入特征的Stacking：{rmse(stkr_pt):.3f}')

abr = AdaBoostRegressor(estimator=stump_reg, n_estimators=100,
    learning_rate=1.5, loss='square', random_state=0)
abr.fit(reg_X_train, reg_y_train)
print(f'AdaBoost：{rmse(abr):.3f}')