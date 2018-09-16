import tensorflow as tf
import numpy as np
from data_loader import DataLoader
import itertools
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
import argparse
import matplotlib.pyplot as plt


time_step = 28
input_size = 28
target_size = 10
learning_rate =	0.001
epoch = 10
batch_size = 32

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test' , action="store_true",help='To test the data')
parser.add_argument('--train', action="store_true",help='To train the data')
parser.add_argument('--hidden_unit', action="store",help='Use model for entered hidden unit size',type=int)
parser.add_argument('--model', action="store",help='Select model',type=str)
args = parser.parse_args()

class LSTMCell_test(rnn.RNNCell):
	tf.set_random_seed(42)
	def __init__(self, num_neurons, input_size, parameters):
		self._num_neurons = num_neurons
		self._input_size  = input_size
		self._parameters = parameters
	
	@property
	def input_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._num_neurons

	@property
	def state_size(self):
		return 2 * self._num_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			IG_Wx = self._parameters['IG_Wx']
			IG_Uh = self._parameters['IG_Uh']
			IG_b = self._parameters['IG_b']

			FG_Wx = self._parameters['FG_Wx']
			FG_Uh = self._parameters['FG_Uh']
			FG_b = self._parameters['FG_b']

			tanh_Wx = self._parameters['tanh_Wx']
			tanh_Uh = self._parameters['tanh_Uh']
			tanh_b = self._parameters['tanh_b']

			OG_Wx = self._parameters['OG_Wx']
			OG_Uh = self._parameters['OG_Uh']
			OG_b = self._parameters['OG_b']
	
			input_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, IG_Wx), tf.matmul(prev_h, IG_Uh) ) , IG_b))
			output_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, OG_Wx), tf.matmul(prev_h, OG_Uh) ) , OG_b))
			forget_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, FG_Wx), tf.matmul(prev_h, FG_Uh) ) , FG_b))
			tanh_output = tf.tanh(tf.add( tf.add( tf.matmul(x, tanh_Wx), tf.matmul(prev_h, tanh_Uh) ) , tanh_b))
			C_t = tf.add(tf.multiply(forget_gate, prev_C), tf.multiply(input_gate, tanh_output))
			y_t = tf.multiply(output_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class LSTMCell_train(rnn.RNNCell):
	tf.set_random_seed(42)
	def __init__(self, num_neurons, input_size):
		self._num_neurons = num_neurons
		self._input_size  = input_size


	@property
	def input_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._num_neurons

	@property
	def state_size(self):
		return 2 * self._num_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_C, prev_h = tf.split(state, 2, 1)
			IG_Wx = tf.get_variable("IG_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			IG_Uh = tf.get_variable("IG_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))
			IG_b  = tf.get_variable("IG_b", [1, self._num_neurons], initializer=tf.ones_initializer())

			FG_Wx = tf.get_variable("FG_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			FG_Uh = tf.get_variable("FG_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))
			FG_b  = tf.get_variable("FG_b", [1, self._num_neurons], initializer=tf.ones_initializer())

			tanh_Wx = tf.get_variable("tanh_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			tanh_Uh = tf.get_variable("tanh_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))
			tanh_b  = tf.get_variable("tanh_b", [1, self._num_neurons], initializer=tf.ones_initializer())

			OG_Wx = tf.get_variable("OG_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			OG_Uh = tf.get_variable("OG_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))
			OG_b  = tf.get_variable("OG_b", [1, self._num_neurons], initializer=tf.ones_initializer())

			input_gate  = tf.sigmoid(tf.add( tf.add( tf.matmul(x, IG_Wx), tf.matmul(prev_h, IG_Uh) ) , IG_b))
			forget_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, FG_Wx), tf.matmul(prev_h, FG_Uh) ) , FG_b))
			output_gate = tf.sigmoid(tf.add( tf.add( tf.matmul(x, OG_Wx), tf.matmul(prev_h, OG_Uh) ) , OG_b))
			tanh_output = tf.tanh(tf.add( tf.add( tf.matmul(x, tanh_Wx), tf.matmul(prev_h, tanh_Uh) ) , tanh_b))
			C_t = tf.add(tf.multiply(forget_gate, prev_C), tf.multiply(input_gate, tanh_output))
			y_t = tf.multiply(output_gate, tf.tanh(C_t))
			return y_t, tf.concat([C_t, y_t], 1)


class GRUCell_test(rnn.RNNCell):
	tf.set_random_seed(42)
	def __init__(self, num_neurons, input_size, parameters):
		self._num_neurons = num_neurons
		self._input_size  = input_size
		self._parameters = parameters
	
	@property
	def input_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._num_neurons

	@property
	def state_size(self):
		return self._num_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			UG_Wx = self._parameters['UG_Wx']
			UG_Uh = self._parameters['UG_Uh']

			RG_Wx = self._parameters['RG_Wx']
			RG_Uh = self._parameters['RG_Uh']

			tanh_Wx = self._parameters['tanh_Wx']
			tanh_Uh = self._parameters['tanh_Uh']
			tanh_b = self._parameters['tanh_b']
	
			update_gate  = tf.sigmoid(tf.add( tf.matmul(x, UG_Wx), tf.matmul(prev_h, UG_Uh) ))
			reset_gate = tf.sigmoid(tf.add( tf.matmul(x, RG_Wx), tf.matmul(prev_h, RG_Uh) ) )
			tanh_output = tf.tanh( tf.add (tf.add( tf.matmul(x, tanh_Wx), tf.matmul(tf.multiply(prev_h, reset_gate), tanh_Uh) ), tanh_b))
			y_t =  tf.add ( tf.multiply((1.0-update_gate),prev_h) , tf.multiply(update_gate, tanh_output) )			
			return y_t, y_t


class GRUCell_train(rnn.RNNCell):
	tf.set_random_seed(42)
	def __init__(self, num_neurons, input_size):
		self._num_neurons = num_neurons
		self._input_size  = input_size


	@property
	def input_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._num_neurons

	@property
	def state_size(self):
		return self._num_neurons


	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			prev_h = state
			UG_Wx = tf.get_variable("UG_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			UG_Uh = tf.get_variable("UG_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))

			RG_Wx = tf.get_variable("RG_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			RG_Uh = tf.get_variable("RG_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))

			tanh_Wx = tf.get_variable("tanh_Wx", [self._input_size, self._num_neurons], initializer=xavier_initializer(seed=42))
			tanh_Uh = tf.get_variable("tanh_Uh", [self._num_neurons, self._num_neurons], initializer=xavier_initializer(seed=42))
			tanh_b  = tf.get_variable("tanh_b", [1, self._num_neurons], initializer=tf.ones_initializer())

			update_gate  = tf.sigmoid(tf.add( tf.matmul(x, UG_Wx), tf.matmul(prev_h, UG_Uh) ))
			reset_gate = tf.sigmoid(tf.add( tf.matmul(x, RG_Wx), tf.matmul(prev_h, RG_Uh) ) )
			tanh_output = tf.tanh( tf.add (tf.add( tf.matmul(x, tanh_Wx), tf.matmul(tf.multiply(prev_h, reset_gate), tanh_Uh) ), tanh_b))
			y_t =  tf.add ( tf.multiply((1.0-update_gate),prev_h) , tf.multiply(update_gate, tanh_output) )			
			return y_t, y_t


def lstm_train(num_neurons, time_step=28, input_size = 28,target_size=10, learning_rate=0.0001,  epoch=100, batch_size=32):

	X = tf.placeholder(tf.float32,	[None,	time_step, input_size])
	y = tf.placeholder(tf.int32,	[None, target_size])  
	# lstm-cell
	current_input = tf.unstack(X , time_step, 1)
	train_lstm = LSTMCell_train(num_neurons, input_size)
	outputs, states	= tf.nn.static_rnn(train_lstm, current_input, dtype=tf.float32)  
	# fully-connected layer
	FC_W = tf.get_variable("FC_W", [num_neurons, target_size], initializer = tf.contrib.layers.xavier_initializer(seed=42))
	FC_b = tf.get_variable("FC_b", [target_size], initializer = tf.zeros_initializer())
	Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
	#optimization
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = y))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	training_op = optimizer.minimize(loss)
	#accuracy
	correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	# data-loading
	ld=DataLoader()
	train_img, train_label =ld.load_data()
	train_label = np.eye(10)[np.asarray(train_label, dtype=np.int32)]
	minibatch_imged, minibatch_labeled = ld.create_batches(train_img, train_label, batch_size)	
	test_img,test_label=ld.load_data(mode='test')
	test_label = np.eye(10)[np.asarray(test_label, dtype=np.int32)]
	test_img = test_img.reshape((-1, time_step, input_size))
	saver = tf.train.Saver()
	weight_filepath = "./weights/lstm/hidden_unit" + str(num_neurons)+ "/model.ckpt"

	with tf.Session() as sess:
		init.run()
		#training
		for epoch in range(epoch):
			for minibatch_img, minibatch_label in itertools.izip(minibatch_imged, minibatch_labeled):
				minibatch_img	= minibatch_img.reshape((-1,	time_step, input_size))
				sess.run(training_op, feed_dict={X:	minibatch_img, y: minibatch_label})	
			acc_train = accuracy.eval(feed_dict={X: minibatch_img, y: minibatch_label})
			print("Train accuracy after %s epochs: %s" %( str(epoch+1), str(acc_train*100) ))
		acc_test = accuracy.eval(feed_dict={X: test_img, y: test_label})				
		#print("Test accuracy:  ", acc_test*100)
		# Save parameters in memory		
		saver.save(sess, weight_filepath)
	return acc_test

def lstm_test(weight_filepath, num_neurons, time_step=28, input_size = 28,target_size=10):

	with tf.Session() as sess:
		X = tf.placeholder(tf.float32,	[None,	time_step, input_size])
		y = tf.placeholder(tf.int32,	[None, target_size])  

		new_saver = tf.train.import_meta_graph(weight_filepath + "/model.ckpt.meta")
		new_saver.restore(sess, tf.train.latest_checkpoint(weight_filepath))
		FC_W    = sess.run('FC_W:0')
		FC_b    = sess.run('FC_b:0') 
		IG_Wx   = sess.run('rnn/LSTMCell_train/IG_Wx:0')
		IG_Uh   = sess.run('rnn/LSTMCell_train/IG_Uh:0')
		IG_b    = sess.run('rnn/LSTMCell_train/IG_b:0')
		FG_Wx   = sess.run('rnn/LSTMCell_train/FG_Wx:0')
		FG_Uh   = sess.run('rnn/LSTMCell_train/FG_Uh:0')
		FG_b    = sess.run('rnn/LSTMCell_train/FG_b:0')
		OG_Wx   = sess.run('rnn/LSTMCell_train/OG_Wx:0')
		OG_Uh   = sess.run('rnn/LSTMCell_train/OG_Uh:0')
		OG_b    = sess.run('rnn/LSTMCell_train/OG_b:0')
		tanh_Wx = sess.run('rnn/LSTMCell_train/tanh_Wx:0')
		tanh_Uh = sess.run('rnn/LSTMCell_train/tanh_Uh:0')
		tanh_b  = sess.run('rnn/LSTMCell_train/tanh_b:0') 
		
		parameters = {
		"IG_Wx" : IG_Wx,
		"IG_Uh" : IG_Uh,
		"IG_b"  : IG_b,
		"FG_Wx" : FG_Wx,
		"FG_Uh" : FG_Uh,
		"FG_b"  : FG_b,
		"tanh_Wx" : tanh_Wx,
		"tanh_Uh" : tanh_Uh,
		"tanh_b" : tanh_b,
		"OG_Wx" : OG_Wx,
		"OG_Uh" : OG_Uh,
		"OG_b"  : OG_b
		}
	
		# lstm-cell
		current_input = tf.unstack(X , time_step, 1)
		lstm_cell_test = LSTMCell_test(num_neurons, input_size, parameters)
		outputs, states	= tf.nn.static_rnn(lstm_cell_test, current_input, dtype=tf.float32)  
		# fully-connected layer
		Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
		correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init = tf.global_variables_initializer()
		# data-loading
		ld=DataLoader()
		test_img,test_label=ld.load_data(mode='test')
		test_label = np.eye(10)[np.asarray(test_label, dtype=np.int32)]
		test_img = test_img.reshape((-1, time_step, input_size))
		acc_test = accuracy.eval(feed_dict={X: test_img, y: test_label})				
		print("Test accuracy:  ", acc_test*100)
		return acc_test


def gru_train(num_neurons, time_step=28, input_size = 28,target_size=10, learning_rate=0.0001,  epoch=100, batch_size=32):
	X = tf.placeholder(tf.float32,	[None,	time_step, input_size])
	y = tf.placeholder(tf.int32,	[None, target_size])  
	# lstm-cell
	current_input = tf.unstack(X , time_step, 1)
	gru_cell_train = GRUCell_train(num_neurons, input_size)
	outputs, states	= tf.nn.static_rnn(gru_cell_train, current_input, dtype=tf.float32)  
	# fully-connected layer
	FC_W = tf.get_variable("FC_W", [num_neurons, target_size], initializer = tf.contrib.layers.xavier_initializer(seed=42))
	FC_b = tf.get_variable("FC_b", [target_size], initializer = tf.zeros_initializer())
	Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
	#optimizatio
	loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = y))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	training_op = optimizer.minimize(loss)
	correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()
	# data-loading
	ld=DataLoader()
	train_img, train_label =ld.load_data()
	train_label = np.eye(10)[np.asarray(train_label, dtype=np.int32)]
	minibatch_imged, minibatch_labeled = ld.create_batches(train_img, train_label, batch_size)	
	test_img,test_label=ld.load_data(mode='test')
	test_label = np.eye(10)[np.asarray(test_label, dtype=np.int32)]
	test_img = test_img.reshape((-1, time_step, input_size))
	saver = tf.train.Saver()
	weight_filepath = "./weights/gru/hidden_unit" + str(num_neurons)+ "/model.ckpt"
	with tf.Session() as sess:
		init.run()
		#training
		for epoch in range(epoch):
			for minibatch_img, minibatch_label in itertools.izip(minibatch_imged, minibatch_labeled):
				minibatch_img	= minibatch_img.reshape((-1,	time_step, input_size))
				sess.run(training_op, feed_dict={X:	minibatch_img, y: minibatch_label})	
			acc_train = accuracy.eval(feed_dict={X: minibatch_img, y: minibatch_label})
			print("Train accuracy after %s epochs: %s" %( str(epoch+1), str(acc_train*100) ))
		acc_test = accuracy.eval(feed_dict={X: test_img, y: test_label})				
		#print("Test accuracy:  ", acc_test*100)
		# Save parameters in memory		
		saver.save(sess, weight_filepath)
	return acc_test


def gru_test(weight_filepath, num_neurons, time_step=28, input_size = 28,target_size=10):
	with tf.Session() as sess:
		X = tf.placeholder(tf.float32,	[None,	time_step, input_size])
		y = tf.placeholder(tf.int32,	[None, target_size])  
		new_saver = tf.train.import_meta_graph(weight_filepath + "/model.ckpt.meta")
		new_saver.restore(sess, tf.train.latest_checkpoint(weight_filepath))
		FC_W    = sess.run('FC_W:0')
		FC_b    = sess.run('FC_b:0') 
		UG_Wx   = sess.run('rnn/GRUCell_train/UG_Wx:0')
		UG_Uh   = sess.run('rnn/GRUCell_train/UG_Uh:0')
		RG_Wx   = sess.run('rnn/GRUCell_train/RG_Wx:0')
		RG_Uh   = sess.run('rnn/GRUCell_train/RG_Uh:0')
		tanh_Wx = sess.run('rnn/GRUCell_train/tanh_Wx:0')
		tanh_Uh = sess.run('rnn/GRUCell_train/tanh_Uh:0')
		tanh_b  = sess.run('rnn/GRUCell_train/tanh_b:0')
		parameters = {
		"UG_Wx" : UG_Wx,
		"UG_Uh" : UG_Uh,
		"RG_Wx" : RG_Wx,
		"RG_Uh" : RG_Uh,
		"tanh_Wx" : tanh_Wx,
		"tanh_Uh" : tanh_Uh,
		"tanh_b" : tanh_b
		}
	
		# lstm-cell
		current_input = tf.unstack(X , time_step, 1)
		gru_cell_test = GRUCell_test(num_neurons, input_size, parameters)
		outputs, states	= tf.nn.static_rnn(gru_cell_test, current_input, dtype=tf.float32)  
		# fully-connected layer
		Z = tf.add(tf.matmul(outputs[-1], FC_W), FC_b)
		correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init = tf.global_variables_initializer()
		# data-loading
		ld=DataLoader()
		test_img,test_label=ld.load_data(mode='test')
		test_label = np.eye(10)[np.asarray(test_label, dtype=np.int32)]
		test_img = test_img.reshape((-1, time_step, input_size))
		acc_test = accuracy.eval(feed_dict={X: test_img, y: test_label})				
		print("Test accuracy:  ", acc_test*100)
		return acc_test



if args.model == 'lstm':    
	if args.train:
		num_neurons = int(args.hidden_unit)
		test_acc = lstm_train(num_neurons, time_step, input_size ,target_size, learning_rate,  epoch, batch_size)
   
	elif args.test:
		num_neurons = int(args.hidden_unit)
		weight_filepath = "./weights/lstm/hidden_unit" + str(num_neurons)
		test_acc = lstm_test(weight_filepath, num_neurons, time_step=28, input_size = 28,target_size=10)

elif args.model == 'gru':    
	if args.train:
		num_neurons = int(args.hidden_unit)
		test_acc = gru_train(num_neurons, time_step, input_size ,target_size, learning_rate,  epoch, batch_size)
   
	elif args.test:
		num_neurons = int(args.hidden_unit)
		weight_filepath = "./weights/gru/hidden_unit" + str(num_neurons)
		test_acc = gru_test(weight_filepath, num_neurons, time_step=28, input_size = 28,target_size=10)

