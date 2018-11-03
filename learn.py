'''
	Author: Dhruv Agarwal
	Course: Introduction to Machine Learning
	Submission Date: 29th September, 2018

	Credits: Logic discussed with Preetha Datta and Prakhar Jain.
'''

from math import exp
from random import uniform
import matplotlib.pyplot as plt
import sys

class NeuralNetwork(object):

	def __init__(self, n, n_tilde, m):
		# n -> number of inputs
		# n_tilde -> number of hidden layer neurons
		# m -> number of outputs

		self.n = n
		self.n_tilde = n_tilde
		self.m = m
		self.ita = 0.03

	# define some utility functions
	def f(self, S):
		'''
			This is the binary sigmoid activation function.
		'''
		ans = 1. / (1. + exp(-10. * S))
		return ans

	def f_prime(self, S):
		'''
			Derivative of the binary sigmoid activation function.
		'''
		y_hat = self.f(S)
		ans = y_hat * (1. - y_hat)
		return ans

	def error_function(self, y, y_hat):
		'''
			returns squared error between the two passed parameters
		'''
		return (y - y_hat) ** 2

	def set_training_data(self, data_x, data_y):
		'''
			Set the training data into x and y lists.
		'''
		self.data_x = data_x
		self.data_y = data_y

	def plot_error(self, x_values, y_values):
		'''
			Plots error with respect to number of epochs.
		'''
		plt.figure(figsize=(8, 6))
		plt.plot(x_values, y_values)
		plt.ylabel('Error')
		plt.xlabel('Number of epochs')
		plt.title('No. neurons in hidden layer: ' + str(self.n_tilde))

		plt.show()

	def __init_weights(self):
		'''
			Set the weights as random values.
			Only to be called once.

			w -> hidden to input weights
			W -> output to hidden weights

			Private method -- cannot be called from outside class definition
		'''
		n = self.n
		n_tilde = self.n_tilde
		m = self.m

		W = []
		for i in range(m + 1):
			temp_W = []
			for j in range(n_tilde + 1):
				if i == 0:
					temp_W.append(None) # to make it 1-indexed
				else:
					random_number = uniform(-0.2, 0.2)
					temp_W.append(random_number)
			W.append(temp_W)

		w = []
		for j in range(n_tilde + 1):
			temp_w = []
			for k in range(n + 1):
				if j == 0:
					temp_w.append(None) # to make it 1-indexed
				else:
					random_number = uniform(-0.2, 0.2)
					temp_w.append(random_number)
			w.append(temp_w)

		return w, W

	def forward_pass(self, data_x, w, W, l):
		'''
			At the end of forward pass, we must have h_j's and y_hat_i's.
			To calculate those, we basically need the weighted sums, and pass those weighted sums through the activation function.

			Needs to be passed the data as well because it makes use of it to calculate y_hat.
		'''
		n = self.n
		n_tilde = self.n_tilde
		m = self.m

		h = [1] # to account for the bias input on the hidden layer
		S_j = [0]
		for j in range(1, n_tilde + 1):
			S = 0
			for k in range(n + 1):
				S += w[j][k] * data_x[l][k]
			S_j.append(S)
			h.append(self.f(S))

		y_hat = [None] # because we want y_hat_1, and not y_hat_0
		S_i = [0]
		for i in range(1, m + 1):
			S = 0
			for j in range(n_tilde + 1):
				S += W[i][j] * h[j]
			S_i.append(S)
			y_hat.append(self.f(S))

		return S_i, S_j, h, y_hat

	def backward_pass(self, S_i, S_j, y_hat, W, l):
		'''
			Go backward (i.e. back propogation) to change the weights.
			For changing weights, we need delta_i and delta_j, which we calculate in this function.
		'''

		n = self.n
		n_tilde = self.n_tilde
		m = self.m
		data_y = self.data_y

		# calculate delta_i's
		del_i = [None]
		for i in range(1, m + 1):
			delta = (data_y[l][i - 1] - y_hat[i]) * self.f_prime(S_i[i])

			del_i.append(delta)

		# calculate delta_j's
		del_j = [None]
		for j in range(1, n_tilde + 1):
			delta = None
			for i in range(1, 1 + m):
				delta = del_i[i] * W[i][j]
			delta *= self.f_prime(S_j[j])
			del_j.append(delta)

		return del_i, del_j

	def train(self, stopping_criteria, plot_J):
		'''
			Main training function that calls all other functions.
			Takes in error argument -- error at which learning should stop.
			Takes another boolean argument which specifies whether a plot of J w.r.t. epochs is required or not.
		'''
		n = self.n
		n_tilde = self.n_tilde
		m = self.m
		ita = self.ita

		w, W = self.__init_weights()
		J = 10 # set high to fulfill while loop enter condition

		data_x = self.data_x
		data_y = self.data_y

		plot_x_axis = []
		plot_y_axis = []

		epochs = 0

		while J > stopping_criteria:
		# while True:
			for l in range(len(data_x)):
				S_i, S_j, h, y_hat = self.forward_pass(data_x, w, W, l)

				del_i, del_j = self.backward_pass(S_i, S_j, y_hat, W, l)

				# main weight update for W - output to hidden
				for i in range(1, m + 1):
					for j in range(n_tilde + 1):
						W[i][j] += ita * del_i[i] * h[j]

				# main weight update for w - hidden to input
				for j in range(1, n_tilde + 1):
					for k in range(n + 1):
						w[j][k] += ita * del_j[j] * data_x[l][k]

			J = 0
			for l in range(len(data_x)):
				S_i, S_j, h, y_hat = self.forward_pass(data_x, w, W, l)

				for i in range(1, m + 1):
					J += self.error_function(data_y[l][i - 1], y_hat[i])

			# print J
			epochs += 1

			if plot_J:
				plot_y_axis.append(J)
				plot_x_axis.append(epochs)

		if plot_J:
			self.plot_error(plot_x_axis, plot_y_axis)

		# set the final weights to the class for future use
		# at this point, this network is trained
		self.W = W
		self.w = w

		return J

	def test(self, test_x, test_y):
		'''
			Tests the trained network on the basis of the testing data provided in the arguments.
		'''
		n = self.n
		n_tilde = self.n_tilde
		m = self.m
		ita = self.ita

		# get the trained weights
		w = self.w
		W = self.W

		J_total = 0
		for l in range(len(test_x)):
			S_i, S_j, h, y_hat = self.forward_pass(test_x, w, W, l)

			# calculate error
			J = 0
			for i in range(1, m + 1):
				J += self.error_function(test_y[l][i - 1], y_hat[i])

			J_total += J

		return J_total

def main():
	# open training data
	with open('hw3data/hw3trainingdata.txt', 'r') as f:
		data_x = []
		data_y = []

		for line in f:
			values = line.strip('\r\n').split(' ')
			data_x.append([1, float(values[1])])
			data_y.append([float(values[2])])

	# take hidden neurons number as CLI input from user
	try:
		hidden_neurons = sys.argv[1]
		hidden_neurons = int(hidden_neurons)
	except Exception as e:
		print 'CLI input required and must be an integer.'
		sys.exit()
	
	network = NeuralNetwork(1, hidden_neurons, 1)
	network.set_training_data(data_x, data_y)
	train_error = network.train(stopping_criteria = 0.04, plot_J = True)

	# open testing data
	with open('hw3data/hw3testingdata.txt', 'r') as f:
		test_x = []
		test_y = []

		for line in f:
			values = line.strip('\r\n').split(' ')
			test_x.append([1, float(values[1])])
			test_y.append([float(values[2])])

	test_error = network.test(test_x, test_y)

	print "Training error:", train_error
	print "Total testing error: ", test_error


if __name__ == '__main__':
	main()

		