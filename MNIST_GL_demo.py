# -*- coding: utf-8 -*-
# Author: E Zhenqian
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global alpha, beta
global num_epoch, num_step
num_epoch = 5
num_step = 5000
alpha = 0.1
beta = 8


def count_neurons(W):
    neurons = tf.math.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(tf.abs(W), axis=1), 10**-3), tf.float32))
    return neurons


def count_sparsity(W):
    count_sum = W.get_shape().as_list()[0] * W.get_shape().as_list()[1]
    count = np.sum(np.int64(np.abs(W.eval()) < 0.001))
    sparsity = float(count / count_sum)
    return sparsity


def norm(W, num):
    if num == 2:
        return tf.reduce_sum(tf.norm(W, axis=1))
    else:
        return tf.reduce_sum(tf.norm(W, ord=1))

def slice_weight(W, num):
    if W.get_shape().as_list()[0] == 1:
        W_new = tf.slice(W, [0, 0], [W.get_shape().as_list()[0], W.get_shape().as_list()[1]-num])
    else:
        W_new = tf.slice(W, [0, 0], [W.get_shape().as_list()[0]-num, W.get_shape().as_list()[1]])
    return W_new

def group_regularization(v):
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), norm(W, 2)) for W in v if 'bias' not in W.name])


def accelerated_GL(v):
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(slice_weight(W, int(W.get_shape().as_list()[0] / beta))),
                                      norm(slice_weight(W, int(W.get_shape().as_list()[0] / beta)), 2))
                          for W in v if 'bias' not in W.name])


def sparse_group_lasso(v):
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    a = tf.reduce_sum([tf.multiply(1 - alpha, tf.multiply(const_coeff(W), norm(W, 2)))
                       for W in v if 'bias' not in W.name])
    b = tf.reduce_sum([tf.multiply(alpha, norm(W, 1)) for W in v if 'bias' not in W.name])
    return tf.add(a, b)


def accelerated_SGL(v):
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    a = tf.reduce_sum([tf.multiply(const_coeff(slice_weight(W, int(W.get_shape().as_list()[0] / beta))),
                                   norm(slice_weight(W, int(W.get_shape().as_list()[0] / beta)), 2))
                       for W in v if 'bias' not in W.name])
    a = tf.multiply(1 - alpha, a)
    b = tf.reduce_sum([tf.multiply(alpha, norm(slice_weight(W, int(W.get_shape().as_list()[0] / beta)), 1))
                       for W in v if 'bias' not in W.name])
    return tf.add(a, b)


def weighted_SGL(v):
    weighted_matrix = tf.diag([0.1, 0.133, 0.167, 0.2])  # lambda_l
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    a = tf.reduce_sum([tf.multiply(weighted_matrix, tf.multiply(const_coeff(W), norm(W, 2)))
                       for W in v if 'bias' not in W.name])
    a = tf.multiply(1 - alpha, a)
    b = tf.reduce_sum([tf.multiply(weighted_matrix, norm(W, 1)) for W in v if 'bias' not in W.name])
    b = tf.multiply(alpha, b)
    return tf.add(a, b)


def accelerated_weighted_SGL(v):
    weighted_matrix = tf.diag([0.1, 0.133, 0.167, 0.2])  # lambda_l
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    a = tf.reduce_sum([tf.multiply(weighted_matrix, tf.multiply(
        const_coeff(slice_weight(W, int(W.get_shape().as_list()[0] / beta))),
        norm(slice_weight(W, int(W.get_shape().as_list()[0] / beta)), 2)))
                       for W in v if 'bias' not in W.name])
    a = tf.multiply(1 - alpha, a)
    b = tf.reduce_sum([tf.multiply(weighted_matrix, norm(slice_weight(W, int(W.get_shape().as_list()[0] / beta)), 1))
         for W in v if 'bias' not in W.name])
    b = tf.multiply(alpha, b)
    return tf.add(a, b)


def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    '''
    image = mnist.train.images[2, :]
    image = image.reshape(28, 28)

    plt.figure()
    plt.imshow(image)
    plt.show()
    '''

    # Reset everything
    tf.reset_default_graph()

    # The directory to save TensorBoard summaries
    from datetime import datetime
    now = datetime.now()
    logdir = "MNIST_summaries/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    x = tf.placeholder(tf.float32, [None, 784])
    # x_image = tf.reshape(X, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    training = tf.placeholder_with_default(False, shape=(), name='training')

    # Helper function to generate a layer
    def create_relu_layer(in_var, in_size, out_size):
        # Parameters for input-hidden layer
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1, seed=80004), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='bias')
        # Output of the hidden layer
        return tf.nn.relu(tf.matmul(in_var, W) + b)

    # Helper function to generate a layer
    def create_softmax_layer(in_var, in_size, out_size):
        # Parameters for input-hidden layer
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1, seed=80004), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='bias')

        # Output of the hidden layer
        return tf.nn.softmax(tf.matmul(in_var, W) + b)

    n = [400, 300, 100, 10]

    with tf.name_scope('hidden_1'):
        h1 = create_relu_layer(x, 784, n[0])
    with tf.name_scope('hidden_2'):
        h2 = create_relu_layer(h1, n[0], n[1])
    with tf.name_scope('hidden_3'):
        h3 = create_relu_layer(h2, n[1], n[2])
    with tf.name_scope('output'):
        y = create_softmax_layer(h3, n[2], n[3])

    # Helper function to check how many neurons are left in a layer
    count_Neurons = lambda W: tf.reduce_sum(tf.cast(tf.greater(
        tf.reduce_sum(tf.abs(W), reduction_indices=[1]), 10**-3), tf.float32))

    # Helper function to calculate the sparsity of each layer
    count_Sparsity = lambda W: tf.subtract(tf.cast(1, tf.float32),
                                           tf.math.divide(tf.reduce_sum(tf.cast(tf.greater(tf.abs(W), 10**-3), tf.float32)),
                                           W.get_shape().as_list()[0] * W.get_shape().as_list()[1]))

    # Get all trainable variables except biases
    v = tf.trainable_variables()

    neurons_summary = tf.summary.scalar('neurons',
                                        tf.reduce_sum([count_neurons(W) for W in v if 'bias' not in W.name]))

    # Define the error function
    with tf.name_scope('cross_entropy_loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # Compute the regularization term
    with tf.name_scope('group_regularization'):
        GL_loss = 0.0001 * group_regularization(v)

    with tf.name_scope('accelerated_GL'):
        aGL_loss = (beta/(beta-1)) * 0.0001 * accelerated_GL(v)

    with tf.name_scope('sparse_group_lasso'):
        SGL_loss = (beta/(beta-1)) * 0.0001 * sparse_group_lasso(v)

    with tf.name_scope('weighted_SGL'):
        wSGL_loss = 0.0001 * weighted_SGL(v)

    with tf.name_scope('accelerated_SGL'):
        aSGL_loss = 0.0001 * accelerated_SGL(v)

    with tf.name_scope('accelerated_weighted_SGL'):
        awSGL_loss = 0.0001 * accelerated_weighted_SGL(v)

    # We attach a logger to the error loss and the regularization part
    loss_summary = tf.summary.scalar('loss', loss)
    GL_loss_summary = tf.summary.scalar('GL_loss', GL_loss)
    aGL_loss_summary = tf.summary.scalar('aGL_loss', aGL_loss)
    SGL_loss_summary = tf.summary.scalar('SGL_loss', SGL_loss)
    wSGL_loss_summary = tf.summary.scalar('wSGL_loss', wSGL_loss)
    aSGL_loss_summary = tf.summary.scalar('aSGL_loss', aSGL_loss)
    awSGL_loss_summary = tf.summary.scalar('waSGL_loss', awSGL_loss)

    #Merge summaries and write them in output
    #merged = tf.summary.merge([loss_summary, neurons_summary])
    #merged = tf.summary.merge([loss_summary, GL_loss_summary, neurons_summary]) # group lasso
    merged = tf.summary.merge([loss_summary, aGL_loss_summary, neurons_summary]) # accelerated group lasso
    #merged = tf.summary.merge([loss_summary, SGL_loss_summary, neurons_summary])
    #merged = tf.summary.merge([loss_summary, aSGL_loss_summary, neurons_summary])
    #merged = tf.summary.merge([loss_summary, wSGL_loss_summary, neurons_summary])
    #merged = tf.summary.merge([loss_summary, awSGL_loss_summary, neurons_summary])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the summary writer
        train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        #start_time = time.time()
        with tf.name_scope('train'):
            #Training function
            #train_step = tf.train.AdamOptimizer().minimize(loss)
            #train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, GL_loss))
            train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, aGL_loss))
            #train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, SGL_loss))
            #train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, aSGL_loss))
            #train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, wSGL_loss))
            #train_step = tf.train.AdamOptimizer().minimize(tf.add(loss, awSGL_loss))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        for epoch in range(num_epoch):
            total_loss = 0.0
            #avg_loss = 0.0
            for i in range(num_step):
                batch_xs, batch_ys = mnist.train.next_batch(400, shuffle=False)
                summary, _, loss_value = sess.run([merged, train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, i)
                total_loss += np.mean(loss_value)
            avg_loss = total_loss / float(num_step)
            print("epoch: %d  loss: %f" % (epoch + 1,  avg_loss))
        # print(sess.run('hidden_1/W:0'))
        # print(sess.run('hidden_2/W:0'))
        duration = time.time() - start_time
        print('The total running time:', duration)
        print('Neurons', sess.run([count_Neurons(W) for W in v if 'bias' not in W.name]))
        print('Neurons', sess.run([count_neurons(W) for W in v if 'bias' not in W.name]))
        print('Sparsity', [count_sparsity(W) for W in v if 'bias' not in W.name])
        print('Sparsity', sess.run([count_Sparsity(W) for W in v if 'bias' not in W.name]))
        save_path = saver.save(sess, "./" + logdir + "model.ckpt")
        print("Model saved in path: %s" % save_path)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Final loss on training set: ', sess.run([loss], feed_dict={x: mnist.train.images,
                                                                          y_: mnist.train.labels}))
        print('training accuracy:', sess.run(accuracy, feed_dict={x: mnist.train.images,
                                                                  y_: mnist.train.labels}))

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Final loss on test set: ', sess.run([loss], feed_dict={x: mnist.test.images,
                                                                      y_: mnist.test.labels}))
        print('test accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                              y_: mnist.test.labels}))

    train_writer.flush()
    train_writer.close()
    '''
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess = tf.InteractiveSession()
    # Train
    tf.global_variables_initializer().run()
    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    print('Final loss on test set: ', sess.run([loss], feed_dict={x: batch_xs, y_: batch_ys}))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
    '''


if __name__ == '__main__':
    main()
