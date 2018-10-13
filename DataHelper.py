import tensorflow as tf
import os
import Augmentor

def augment(dataPath) :
	p = Augmentor.Pipeline("data/payingAttention/")

	p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
	p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

	p.sample(1000)


def getImages(dataPath, batchSize, height=64, width=64,inputChannels=3):

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