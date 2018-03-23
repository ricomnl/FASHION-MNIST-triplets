import tensorflow as tf
import numpy as np
import math
from tensorflow.python.framework import ops
from random import shuffle

def get_n_random_images(anchors, i):
    list = np.arange(10)
    list = np.delete(list, i)
    negative = np.empty(anchors[i].shape, dtype = np.float32)
    for k in range(len(anchors[i])):
        c = np.random.choice(list)
        negative[k] = anchors[c][np.random.choice(anchors[c].shape[0])]
    
    return negative

def create_triplets(X, Y):
    anchors = [X[np.where(Y==i)[0]] for i in np.unique(Y)]
    positive = []
    negative = []
    triplets = []
    #create positive list
    for i in np.unique(Y):
        positive.append(np.take(anchors[i], np.random.permutation(anchors[i].shape[0]), axis=0))
    #create negative list
    for j in np.unique(Y):
        negative.append(get_n_random_images(anchors, j))
    #merge as triplets
    for t in np.unique(Y):
         for t2 in range(anchors[t].shape[0]):
            cache = (anchors[t][t2], positive[t][t2], negative[t][t2])
            triplets.append(cache)
            
    return triplets
        
def list_to_array(triplets):
    anchors = np.empty([len(triplets), triplets[0][0].shape[0], triplets[0][0].shape[1], triplets[0][0].shape[2]])
    positive = np.empty([len(triplets), triplets[0][0].shape[0], triplets[0][0].shape[1], triplets[0][0].shape[2]])
    negative = np.empty([len(triplets), triplets[0][0].shape[0], triplets[0][0].shape[1], triplets[0][0].shape[2]])
    for i in range(len(triplets)):
        anchors[i] = triplets[i][0]
        positive[i] = triplets[i][1]
        negative[i] = triplets[i][2]
        
    return anchors, positive, negative
        
def random_mini_batches(triplets, mini_batch_size):
    m = len(triplets)
    mini_batches = []
    
    #Step 1: Shuffle
    shuffle(triplets)
    
    #Step 2: Partition
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch = triplets[k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batches.append(mini_batch)

    #Handling the end case
    if m % mini_batch_size != 0:
        mini_batch = triplets[: m - mini_batch_size * num_complete_minibatches]
        mini_batches.append(mini_batch)
        
    return mini_batches

def create_placeholders(n_H0, n_W0, n_C0):
    A = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
    P = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
    N = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
    
    return A, P, N

def initialize_parameters():
    W1 = tf.get_variable("W1", [3,3,1,8], initializer = tf.contrib.layers.xavier_initializer())
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [8]))
    W2 = tf.get_variable("W2", [3,3,8,16], initializer = tf.contrib.layers.xavier_initializer())
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [16]))
    W3 = tf.get_variable("W3", [5,5,16,16], initializer = tf.contrib.layers.xavier_initializer())
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [16]))
    W4 = tf.get_variable("W4", [3,3,16,32], initializer = tf.contrib.layers.xavier_initializer())
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    parameters = {"W1" : W1, "W2": W2, "W3": W3, "W4": W4, "B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5}
    
    return parameters

def feature_generation(X, parameters, dropout = 0.6):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
    B1 = parameters["B1"]
    B2 = parameters["B2"]
    B3 = parameters["B3"]
    B4 = parameters["B4"]
    B5 = parameters["B5"]
    
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'VALID')
    Z1 = tf.nn.batch_normalization(Z1, mean = 0, variance = 1, offset = B1, scale = None, variance_epsilon = 1e-5)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.nn.conv2d(A1, W2, strides = [1,1,1,1], padding = 'VALID')
    Z2 = tf.nn.batch_normalization(Z2, mean = 0, variance = 1, offset = B2, scale = None, variance_epsilon = 1e-5)
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    Z3 = tf.nn.conv2d(P2, W3, strides = [1,1,1,1], padding = 'VALID')
    Z3 = tf.nn.batch_normalization(Z3, mean = 0, variance = 1, offset = B3, scale = None, variance_epsilon = 1e-5)
    A3 = tf.nn.relu(Z3)
    
    Z4 = tf.nn.conv2d(A3, W4, strides = [1,1,1,1], padding = 'VALID')
    Z4 = tf.nn.batch_normalization(Z4, mean = 0, variance = 1, offset = B4, scale = None, variance_epsilon = 1e-5)
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    P4 = tf.contrib.layers.flatten(P4)
    Z5 = tf.contrib.layers.fully_connected(P4, 32, activation_fn = None)
    Z5 = tf.nn.dropout(Z5, dropout)
    Z5 = tf.nn.batch_normalization(Z5, mean = 0, variance = 1, offset = B5, scale = None, variance_epsilon = 1e-5)
    Z6 = tf.contrib.layers.fully_connected(Z5, 32)
    
    return Z6

def triplet_loss(A, P, N, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.squared_difference(A, P), axis = -1)
    neg_dist = tf.reduce_sum(tf.squared_difference(A, N), axis = -1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

def model(X_train, Y_train, X_cv, Y_cv, learning_rate = 0.001, num_epochs = 100, minibatch_size = 64, drop_conv = 0.8, alpha = 0.2, features = 32, print_cost = True):
    ops.reset_default_graph()
    (m, n_H0, n_W0, n_C0) = X_train.shape
    costs = []
    triplets = create_triplets(X_train, Y_train)
    
    A, P, N = create_placeholders(n_H0, n_W0, n_C0)
    parameters = initialize_parameters()
    f_A = feature_generation(A, parameters)
    f_P = feature_generation(P, parameters)
    f_N = feature_generation(N, parameters)    
    loss = triplet_loss(f_A, f_P, f_N, alpha = alpha)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        print("Training started...")
        for epoch in range(num_epochs):
            minibatch_loss = 0.
            num_minibatches = int(m/ minibatch_size)
            minibatches = random_mini_batches(triplets, minibatch_size)
            
            for minibatch in minibatches:
                minibatch_A, minibatch_P, minibatch_N = list_to_array(minibatch)
                _, temp_loss = sess.run([optimizer, loss], feed_dict = {A: minibatch_A, P: minibatch_P, N: minibatch_N})
                
                minibatch_loss += temp_loss / num_minibatches
                
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_loss))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_loss)
        
        features = f_A.eval({A: X_train})
        
        save_path = saver.save(sess, "model/model.ckpt")
        print("Model saved in path: %s" % save_path)
        return features