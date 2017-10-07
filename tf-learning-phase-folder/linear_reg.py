import tensorflow as tf 

theta0 = tf.Variable([1.], tf.float32)
theta1 = tf.Variable([1.], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# linear model
h = theta0+theta1*x
# loss function
J = tf.reduce_sum(tf.square(h-y))
# optimizer defined
alpha = 0.01
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(J)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
	sess.run(train, {x:x_train,y:y_train})

curr_theta0, curr_theta1, curr_J = sess.run([theta0, theta1, J],{x:x_train, y:y_train})
print ("Theta0: %s Theta1: %s loss: %s"%(curr_theta0, curr_theta1, curr_J))