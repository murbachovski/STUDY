import tensorflow as tf


x_train = [1,2,3] # 1
y_train = [1,2,3] # 2
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

hypothesis = x * w
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse


lr = 1
gradient = tf.reduce_mean((x * w - y) * x)
# gradient = tf.reduce_mean((hypothesis - y) * x)

descent = w - lr * gradient
update = w.assign(descent)  # w = w - lr * gradient

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2):

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ W_history')
print(w_history)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ loss_history')
print(loss_history)

# plt.scatter(w_history, loss_history)
# plt.xlabel('Weight')
# plt.ylabel('Loss')
# plt.show()

