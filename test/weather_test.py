'''
Created on 2018. 1. 10.

@author: acorn
'''
import tensorflow as tf
import numpy as np

data = np.loadtxt("2012_2015new.csv", dtype=np.float32, delimiter = ',')

print(data.shape)

table_col = data.shape[1]

column = table_col - 1

x_data = data[:, 0:column]
y_data = data[:, column:(column+1)]

print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape = [None, column])
y = tf.placeholder(tf.float32, shape = [None, 1])
w = tf.Variable(tf.random_normal([column, 1]))
b = tf.Variable(0.0)

H = tf.matmul(x, w) + b

diff = tf.square(H - y)
cost = tf.reduce_mean(diff)

learn_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5000):
    _t, _w, _c, _h = sess.run([train, w, cost, H], feed_dict={x : x_data, y : y_data})
    if step % 100 == 0:
        print("step :%d, loss:%f" % (step, _c))
    
# result = sess.run(H, feed_dict={x : x_test})
result = sess.run( H, feed_dict={ x : x_data })
print('score predict', result) # 예측 결과