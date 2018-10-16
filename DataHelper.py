import tensorflow as tf
import os
import Augmentor
import cv2
import numpy as np
from keras.utils import np_utils


def augment(dataPath) :
	p = Augmentor.Pipeline(dataPath)

	#p.rotate(probability=0.4, max_left_rotation=5, max_right_rotation=5)
	p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
	p.random_brightness(0.6, 0.6, 1.4)

	p.sample(1024)


def getImagesFromDirectory(dataPath, batchSize, height=64, width=64,inputChannels=3):

    imagePaths, labels = list(), list()

    # An ID will be affected to each sub-folders by alphabetical order
    label = 0

    # List the directory
    classes = sorted(os.walk(dataPath).__next__()[1])


    # List each sub-directory (the classes)
    for c in classes:
        classDir = os.path.join(dataPath, c)
        iteration = os.walk(classDir).__next__()

        # Add each image to the training set
        for sample in iteration[2]:
            imagePaths.append(os.path.join(classDir, sample))
            labels.append(label)
        label += 1

    # Convert to Tensor
    imagePaths = tf.convert_to_tensor(imagePaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    #labels = tf.one_hot(labels, 2)

    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagePaths, labels], shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=inputChannels)

    # Resize images to a common size
    image = tf.image.resize_images(image, [height, width])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batchSize, capacity=batchSize * 8, num_threads=4)

    return X, Y

def loadImagesFromFile(dataset_path, batch_size, IMG_HEIGHT=160, IMG_WIDTH=320) : 

	imagepaths, labels = list(), list()

	data = open(dataset_path, 'r').read().splitlines()
	for d in data:
		imagepaths.append(d.split(',')[0])
		labels.append(int(d.split(',')[3]))

	imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
	labels = tf.convert_to_tensor(labels, dtype=tf.float32)

	# Build a TF Queue, shuffle data
	image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)

	# Read images from disk
	image = tf.read_file(image)
	image = tf.image.decode_jpeg(image, channels=3)

	image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

	# Create batches
	X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)

	return X, Y

def imagesToNdarray(dataPath, binary = 'y'):
	imagePaths, labels = list(), list()
	label = 0

	classes = sorted(os.walk(dataPath).__next__()[1])


	# List each sub-directory (the classes)
	for c in classes:
		classDir = os.path.join(dataPath, c)
		iteration = os.walk(classDir).__next__()

		# Add each image to the training set
		for sample in iteration[2]:
			imagePaths.append(os.path.join(classDir, sample))
			labels.append(label)
		label += 1

	#r_state = np.random.get_state()

	#np.random.shuffle(imagePaths)

	X = np.ndarray((2048, 64, 64,3), dtype=np.float32)

	i = 0
	for image in imagePaths:
		im = cv2.imread(image,1)
		im = cv2.resize(im, (64,64))
		im = im / 255
		X[i] = im
		i += 1


	#X.reshape(X.shape + (1,))

	if binary == "y" :
		# Do something
		labels = np.array(labels)
		#np.random.set_state(r_state)
		#np.random.shuffle(labels)
	else:
		labels = np_utils.to_categorical(labels)
		#np.random.set_state(r_state)
		#np.random.shuffle(labels)
	return X, labels

