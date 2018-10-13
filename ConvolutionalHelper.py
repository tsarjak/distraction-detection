import tensorflow as tensorflow
import numpy as np

def initWeights(shape, stdDev = 0.1):
	randomDist = tf.truncated_normal(shape, stddev = stdDev)
	return tf.Variable(randomDist)

def initBias(shape, val = 0.1):
	biasVal = tf.constant(val, shape=shape)
	return tf.Variable(biasVal)

# x -> (batch, Height, Width, Channels)
# W -> (kernal H, kernal W, channels In, channels out)
def conv2d(x, W, strides = [1,1,1,1], padding = 'SAME'):
	return tf.nn.conv2d(x, W, strides=strides, padding=padding)

# x -> [batch, h, w, c]
def maxPool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME'):
	return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding)

def convLayerRelu(inputX, shape):
	W = initWeights(shape)
	b = initBias([shape[3]])
	return tf.nn.relu(conv2d(inputX, W) + b)

def convLayerLeakyRelu(inputX, shape):
	W = initWeights(shape)
	b = initBias([shape[3]])
	return tf.nn.leaky_relu(conv2d(inputX, W) + b)

def fullyConnectLayer(inputLayer, size):
    inputSize = int(inputLayer.get_shape()[1])
    W = initWeights([inputSize, size])
    b = initBias([size])
    return tf.matmul(inputLayer, W) + b