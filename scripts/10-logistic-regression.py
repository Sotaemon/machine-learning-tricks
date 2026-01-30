import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

lines = np.loadtxt('./data/lr_dataset.csv', delimiter=',', dtype=float)
features = lines[:, 0:-1]
labels = lines[:, -1]

print('数据集规模：', len(labels))

pos_index = np.where(labels == 1)
neg_index = np.where(labels == 0)

plt.scatter(features[pos_index, 0], features[pos_index, 1], marker='o', color='coral', s=10)
plt.scatter(features[neg_index, 0], features[neg_index, 1], marker='x', color='blue', s=10 )
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')
plt.show()

np.random.seed(42)
ratio = 0.7
split = int(len(features) * ratio)

index = np.random.permutation(len(features))
features, labels = features[index], labels[index]

train_features = features[:split]
train_labels = labels[:split]
test_features = features[split:]
test_labels = labels[split:]

def acc(label_true, label_pred):
    return np.mean(label_true == label_pred)
def auc(label_true, label_pred):
    index = np.argsort(label_pred)[::-1]
    label_true = label_true[index]
    label_pred = label_pred[index]

    tp = np.cumsum(label_true)
    fp = np.cumsum(1 - label_true)
    
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]

    s = 0.0
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    for i in range(1, len(fpr)):
        s += (fpr[i] - fpr[i - 1]) * tpr[i]
    return s

def logistic(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(num_epochs, learning_rate, weight_decay):
    theta = np.random.normal(size=(train_set.shape[1],))
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []
    for i in range(num_epochs):
        pred = logistic(train_set @ theta)
        grad = - train_set.T @ (train_labels - pred) + weight_decay * theta
        theta -= learning_rate * grad

        train_loss = - train_labels.T @ np.log(pred) - (1 - train_labels).T @ np.log(1-pred) + weight_decay * np.linalg.norm(theta) ** 2 / 2
        train_losses.append(train_loss / len(train_set))
        test_pred = logistic(test_set @ theta)
        test_loss = - test_labels.T @ np.log(test_pred) - (1 - test_labels).T @ np.log(1 - test_pred)
        test_losses.append(test_loss / len(test_set))

        train_acc.append(acc(train_labels, pred >= 0.5))
        test_acc.append(acc(test_labels, test_pred >= 0.5))
        train_auc.append(auc(train_labels, pred))
        test_auc.append(auc(test_labels, test_pred))
    return theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc

num_epochs = 250
learning_rate = 0.002
weight_decay = 1.0
np.random.seed(42)

train_set = np.concatenate([train_features, np.ones((train_features.shape[0], 1))], axis=1)
test_set = np.concatenate([test_features, np.ones((test_features.shape[0], 1))], axis = 1)

theta, train_losses, test_losses, train_acc, test_acc, train_auc, test_auc = gradient_descent(num_epochs, learning_rate, weight_decay)

labels_pred = np.where(logistic(test_set @ theta) >= 0.5, 1, 0)
final_acc = acc(test_labels, labels_pred)

print('预测准确率：', final_acc)
print('回归系数：', theta)

plt.figure(figsize=(13, 9))
xticks = np.arange(num_epochs) + 1
# 绘制训练曲线
plt.subplot(221)
plt.plot(xticks, train_losses, color='blue', label='train loss')
plt.plot(xticks, test_losses, color='red', ls='--', label='test loss')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(222)
plt.plot(xticks, train_acc, color='blue', label='train accuracy')
plt.plot(xticks, test_acc, color='red', ls='--', label='test accuracy')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制AUC
plt.subplot(223)
plt.plot(xticks, train_auc, color='blue', label='train AUC')
plt.plot(xticks, test_auc, color='red', ls='--', label='test AUC')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

# 绘制模型学到的分隔直线
plt.subplot(224)
plot_x = np.linspace(-1.1, 1.1, 100)
# 直线方程：theta_0 * x_1 + theta_1 * x_2 + theta_2 = 0
plot_y = -(theta[0] * plot_x + theta[2]) / theta[1]
pos_index = np.where(labels == 1)
neg_index = np.where(labels == 0)
plt.scatter(features[pos_index, 0], features[pos_index, 1], 
    marker='o', color='coral', s=10)
plt.scatter(features[neg_index, 0], features[neg_index, 1], 
    marker='x', color='blue', s=10)
plt.plot(plot_x, plot_y, ls='-.', color='green')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('X1 axis')
plt.ylabel('X2 axis')
plt.savefig('./output/logistic/lr_dataset.png')
plt.savefig('./output/logistic/lr_dataset.pdf')
plt.show()

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(train_features, train_labels)
print('回归系数：', lr_clf.coef_[0], lr_clf.intercept_)
pred_labels = lr_clf.predict(test_features)
print('准确率为：',np.mean(pred_labels == test_labels))