import random
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 获取数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# print(x_data)
# print(y_data)
#  乱序
random_seed = random.randint(0, 998244353)
print(random_seed)
np.random.seed(random_seed)
np.random.shuffle(x_data)
np.random.shuffle(y_data)

# 划分训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 数据类型转换
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 划分标签对
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络参数，标记为可训练
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_result = []
test_acc = []
epoch = 500
loss_all = 0

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()

        # 求偏导
        grads = tape.gradient(loss, [w1, b1])

        # 更新参数
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
    print("Epoch: {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_result.append(loss_all / 4)
    loss_all = 0

    # 测试部分
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        total_correct = tf.reduce_sum(total_correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print()
plt.title("loss function curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_result, label="$Loss$")
plt.legend()
plt.show()

plt.title("ACC curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()
