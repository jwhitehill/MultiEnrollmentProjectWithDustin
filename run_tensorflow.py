import tensorflow as tf
import common
import util
import numpy as np
import pandas

BATCH_SIZE = 50

def makeLabels (y):
	labels = np.hstack((1 - np.atleast_2d(y).T, np.atleast_2d(y).T)).astype(np.float64)
	return labels

if __name__ == "__main__":
	d = util.loadTrainingSet()
	fields = list(d.columns)
	fields.remove(common.TARGET_VARIABLE)

	data_y = d[common.TARGET_VARIABLE].as_matrix() > 1
	data_x = d[fields].as_matrix().astype(np.float32)

	data_y = data_y[0:1000]
	data_x = data_x[0:1000,:]

	# Scale data
	mx = np.mean(data_x, axis=0)
	data_x -= np.tile(np.atleast_2d(mx), (data_x.shape[0], 1))
	sx = np.std(data_x, axis=0)
	sx[sx == 0] = 1
	data_x /= np.tile(np.atleast_2d(sx), (data_x.shape[0], 1))

	session = tf.InteractiveSession()
	x = tf.placeholder("float", shape=[None, data_x.shape[1]])
	y_ = tf.placeholder("float", shape=[None, 2])

	W1 = tf.Variable(tf.truncated_normal([data_x.shape[1],10], stddev=0.01))
	b1 = tf.Variable(tf.truncated_normal([10], stddev=0.01))
	W2 = tf.Variable(tf.truncated_normal([10,2], stddev=0.01))
	b2 = tf.Variable(tf.truncated_normal([2], stddev=0.01))

	z = tf.nn.relu(tf.matmul(x,W1) + b1)
	y = tf.nn.softmax(tf.matmul(z,W2) + b2)

	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	#train_step = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.1).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(cross_entropy)

	session.run(tf.initialize_all_variables())
	for i in range(10000):
		offset = i*BATCH_SIZE % (data_x.shape[0] - BATCH_SIZE)
		train_step.run({x: data_x[offset:offset+BATCH_SIZE, :], y_: makeLabels(data_y[offset:offset+BATCH_SIZE])})
		if i % 100 == 0:
			print cross_entropy.eval({x: data_x, y_: makeLabels(data_y)})
