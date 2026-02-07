import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

input_path = "./data/movielens_100k.csv"
output_path = "./output/bilinear"
ratio = 0.8

data = np.loadtxt(input_path, delimiter=",", dtype=int)
print("=" * 50)
print("Dataset Size: ", len(data))
print("Data Shape (Original):\n", data[:5], "\n...")


data[:, :2] = data[:, :2] - 1

users = set()
items = set()
for i, j, k in data:
    users.add(i)
    items.add(j)
user_num = len(users)
item_num = len(items)
print("=" * 50)
print("Users: ", user_num, "Items: ", item_num)

np.random.shuffle(data)
np.random.seed(42)
split = int(len(data) * ratio)
train, test = data[:split], data[split:]
print("Data Shape (Randomized):\n", data[:5], "\n...")

user_cnt = np.bincount(train[:, 0], minlength=user_num)
item_cnt = np.bincount(train[:, 1], minlength=item_num)

user_train, user_test = train[:, 0], test[:, 0]
item_train, item_test = train[:, 1], test[:, 1]
label_train, label_test = train[:, 2], test[:, 2]


class matrix_factorization:
    pass
