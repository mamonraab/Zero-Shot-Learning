import tensorflow as tf
import numpy as np 

v1 = tf.Variable(tf.random_normal(shape=[2,5]), name='v1')
v2 = tf.Variable(tf.random_normal(shape=[5,7]), name='v2')

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.save(sess, 'my-first-model-save')