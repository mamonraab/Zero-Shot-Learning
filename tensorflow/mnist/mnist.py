import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

# /tmp/data file will automatically be deleted when the program exits or file is closed.
mnist = input_data.read_data_sets("/tmp/data", one_hot= True)

batch_size = 100

nunits_input = 784
nunits_hidden1 = 500
nunits_hidden2 = 500
nunits_hidden3 = 500
nunits_output = 10

# No. of hidden layers = 3
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

def neural_network_model(data):

	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([nunits_input, nunits_hidden1])),
						'biases':tf.Variable(tf.random_normal([nunits_hidden1]))}

	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nunits_hidden1, nunits_hidden2])),
						'biases':tf.Variable(tf.random_normal([nunits_hidden2]))}

	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nunits_hidden2, nunits_hidden3])),
						'biases':tf.Variable(tf.random_normal([nunits_hidden3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([nunits_hidden3, nunits_output])),
						'biases':tf.Variable(tf.random_normal([nunits_output]))}

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']),hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']),hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']),hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']),output_layer['biases'])

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	sess = tf.Session()
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prediction,logits=y))
	alpha = 0.001
	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		# initializing all variables
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				# sess.run([] ,...) - The thing in [] can only be applied to tensor variables such as
				# train or cost which are defined like tf.something
				_, c = sess.run([train, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print ('Epoch', epoch , 'completed of', hm_epochs, 'cost function:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print ('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)