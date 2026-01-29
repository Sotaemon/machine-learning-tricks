import numpy
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 读取数据
lines = numpy.loadtxt("./data/USA_Housing.csv", delimiter=",", dtype="str")
header = lines[0]
lines = lines[1:].astype(float)

##############################################
print("数据特征：", ", ".join(header[:-1]))
print("数据标签：", header[-1])
print("数据总条数：", len(lines))
##############################################

# 打乱数据
numpy.random.seed(42)
lines = numpy.random.permutation(lines)

# 取 80%
ratio = 0.8
split = int(len(lines) * ratio)
train, test = lines[:split], lines[split:]

# 归一化
scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
# 区分特征值与标签
train_features, train_labels = train[:, :-1], train[:, -1].flatten()
test_features, test_labels = test[:, :-1], test[:, -1].flatten()

# 为特征值增加常数 1
train_set = numpy.concatenate(
    [train_features, numpy.ones((len(train_features), 1))], axis=-1
)

# 矩阵求解法
theta = numpy.linalg.inv(train_set.T @ train_set) @ train_set.T @ train_labels
print("回归系数（θ）：")
for i, coef in enumerate(theta.flatten()):
    print(f" θ{i}:\t{coef:.8f}")

test_set = numpy.concatenate(
    [test_features, numpy.ones((len(test_features), 1))], axis=-1
)
test_pre = test_set @ theta

rmse_loss = numpy.sqrt(numpy.square(test_labels - test_pre).mean())
print("RMSE：", rmse_loss)

# skleanrn 法
linreg = LinearRegression()
linreg.fit(train_features, train_labels)
print("回归系数：", linreg.coef_, linreg.intercept_)
test_pre = linreg.predict(test_features)
rmse_loss = numpy.sqrt(numpy.square(test_labels - test_pre).mean())
print("RMSE：", rmse_loss)


# 梯度下降法
def batch_generator(x, y, batch_size, shuffle=True):
    if shuffle:
        idx = numpy.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    total_samples = len(x)
    current_batch_start = 0
    while current_batch_start < total_samples:
        batch_end = min(current_batch_start + batch_size, total_samples)
        yield x[current_batch_start:batch_end], y[current_batch_start:batch_end]
        current_batch_start = batch_end


def stochastic_gradient_descent(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[numpy.ndarray, list[float], list[float]]:
    # 使用默认值或传入的参数
    matrix_x = train_set
    matrix_y = test_set
    actual_train_labels = train_labels
    actual_test_labels = test_labels

    # 验证输入数据
    if (
        matrix_x is None
        or actual_train_labels is None
        or matrix_y is None
        or actual_test_labels is None
    ):
        raise ValueError("训练数据或测试数据未提供")

    if matrix_x.shape[0] != actual_train_labels.shape[0]:
        raise ValueError(
            f"训练特征和标签数量不匹配: {matrix_x.shape[0]} vs {actual_train_labels.shape[0]}"
        )

    if matrix_y.shape[0] != actual_test_labels.shape[0]:
        raise ValueError(
            f"测试特征和标签数量不匹配: {matrix_y.shape[0]} vs {actual_test_labels.shape[0]}"
        )

    # 初始化权重参数
    theta = numpy.random.normal(size=matrix_x.shape[1])

    # 存储损失值
    train_losses = []
    test_losses = []

    # 训练循环
    for epoch in range(num_epochs):
        # 创建批次生成器
        batch_generator_obj = batch_generator(
            matrix_x, actual_train_labels, batch_size, shuffle=True
        )

        epoch_train_loss = 0

        for x_batch, y_batch in batch_generator_obj:
            # 计算梯度
            predictions = x_batch @ theta
            residuals = predictions - y_batch
            gradient = x_batch.T @ residuals

            # 更新参数
            theta = theta - learning_rate * gradient / len(x_batch)

            # 累积批次损失
            batch_loss = numpy.square(residuals).sum()
            epoch_train_loss += batch_loss

        # 计算平均训练损失（均方根误差）
        avg_train_loss = numpy.sqrt(epoch_train_loss / len(matrix_x))
        train_losses.append(avg_train_loss)

        # 计算测试损失
        test_predictions = matrix_y @ theta
        test_residuals = test_predictions - actual_test_labels
        test_loss = numpy.sqrt(numpy.square(test_residuals).mean())
        test_losses.append(test_loss)

        # 可选：打印进度
        # if (epoch + 1) % max(1, num_epochs // 10) == 0:
        #    print(
        #        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}"
        #    )

    print("回归系数：", theta)
    return theta, train_losses, test_losses


num_epoch = 20
learning_rate = 0.1
batch_size = 32

numpy.random.seed(42)

_, train_losses, test_losses = stochastic_gradient_descent(
    num_epoch, learning_rate, batch_size
)
pyplot.plot(numpy.arange(num_epoch), train_losses, color="blue", label="train loss")
pyplot.plot(
    numpy.arange(num_epoch), test_losses, color="red", ls="--", label="testloss"
)
pyplot.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
pyplot.xlabel("Epoch")
pyplot.ylabel("RMSE")
pyplot.legend()
pyplot.show()
