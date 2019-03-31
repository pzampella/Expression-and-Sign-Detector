from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
from skimage import exposure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(13)

# Import data
path_train = "/home/pzampella/NimSet/MNIST/train.csv"
path_test = "/home/pzampella/NimSet/MNIST/test.csv"
train_data = np.genfromtxt(path_train, dtype=None, delimiter=',')
test_data = np.genfromtxt(path_test, dtype=None, delimiter=',')
# Parameters
learning_rate = 0.001
training_iters = 1000000
new_dim = 28
conv = 5
batch_size = 128
dropout = 0.2      # Dropout, probability to keep units

# Network Parameters
n_input = new_dim*new_dim # data input (img shape: 28*28)
n_classes = 24 # total classes (alphabet signs without J and Z)

_training = []
_label_training = np.zeros((2*len(train_data), (n_classes)), dtype=int)
_testing = []
_label_testing = np.zeros((2*len(test_data), (n_classes)), dtype=int)
for i in range(1, len(train_data)):
    aux = np.array(map(int, train_data[i][1:(new_dim * new_dim) + 1]))
    aux2 = aux.reshape((28, 28))
    contrast = exposure.equalize_hist(aux2)
    aux3 = np.concatenate(contrast*255)
    _training.append(aux3)
    idx = int(train_data[i][0])
    if idx > 8: #Falta el gesto para la J
        idx -= 1
    _label_training[i-1][idx] = 1 #hot pixel
for i in range(1, len(train_data)):
    aux = np.array(map(int, train_data[i][1:(new_dim * new_dim) + 1]))
    aux2 = np.flip(aux.reshape((28, 28)), 1)
    contrast = exposure.equalize_hist(aux2)
    aux3 = np.concatenate(contrast*255)
    _training.append(aux3)
    idx = int(train_data[i][0])
    if idx > 8: #Falta el gesto para la J
        idx -= 1
    _label_training[i-1+len(train_data)][idx] = 1
for i in range(1, len(test_data)):
    aux = np.array(map(int, test_data[i][1:(new_dim * new_dim) + 1]))
    aux2 = aux.reshape((28, 28))
    contrast = exposure.equalize_hist(aux2)
    aux3 = np.concatenate(contrast*255)
    _testing.append(aux3)
    idx = int(test_data[i][0])
    if idx > 8: #Falta el gesto para la J
        idx -= 1
    _label_testing[i-1][idx] = 1
for i in range(1, len(test_data)):
    aux = np.array(map(int, test_data[i][1:(new_dim * new_dim) + 1]))
    aux2 = np.flip(aux.reshape((28, 28)),1)
    contrast = exposure.equalize_hist(aux2)
    aux3 = np.concatenate(contrast*255)
    _testing.append(aux3)
    idx = int(test_data[i][0])
    if idx > 8: #Falta el gesto para la J
        idx -= 1
    _label_testing[i-1+len(test_data)][idx] = 1

training = np.array(_training)
label_training = np.array(_label_training)
testing = np.array(_testing)
label_testing = np.array(_label_testing)
# print(label_training[0:10])
# print(label_testing[0:10])
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_input])   # 28x28
ys = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, new_dim, new_dim, 1])/255.

## conv1 layer ##
W_conv1 = weight_variable([conv, conv, 1, 32], "W_conv1_h") # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], "b_conv1_h")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([conv, conv, 32, 64], "W_conv2_h") # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], "b_conv2_h")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([(new_dim/4)*(new_dim/4)*64, 1024], "W_fc1_h")
b_fc1 = bias_variable([1024], "b_fc1_h")
h_pool2_flat = tf.reshape(h_pool2, [-1, (new_dim/4)*(new_dim/4)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, n_classes], "W_fc2_h")
b_fc2 = bias_variable([n_classes], "b_fc2_h")
prediction = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
best = 0
aciertos = 0.0

print("Optimizing!")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    min_acc = 55.0
    # Keep training until reach max iterations
    while step < training_iters:
        rand = np.random.randint(0, len(training), size=batch_size)
        sess.run(train_step, feed_dict={xs: training[rand], ys: label_training[rand], keep_prob: dropout})
        step += 1
        if step % 10 == 0:
            acc = compute_accuracy(training[rand], label_training[rand])
            if step % 100 == 0:
                aciertos = 0.0
                rand2 = np.random.randint(0, len(testing), size=batch_size)
                predict = tf.argmax(prediction, 1)
                pred = predict.eval({xs: testing[rand2], ys: np.zeros((batch_size, (n_classes)), dtype=int), keep_prob: 1.0})
                # print(label_testing[0:1])
                # print(label_training[3:4])
                # print(predict.eval({xs: testing[0:1], ys: np.zeros((1, (n_classes)), dtype=int), keep_prob: 1.0}))
                # print(predict.eval({xs: training[3:4], ys: np.zeros((1, (n_classes)), dtype=int), keep_prob: 1.0}))
                for i in range(0, batch_size):
                    if pred[i] == np.argmax(label_testing[rand2[i]]):
                        aciertos += 1.0
                print("Iteration: " + str(step) + r"/" + str(training_iters) + "   Training Accuracy: " + str(acc) + "    Test Accuracy: " + str(100 * aciertos / batch_size) + "%")
            else:
                print("Iteration: " + str(step) + r"/" + str(training_iters) + "   Training Accuracy: " + str(acc))
            if acc == 1.0: #Early stop to avoid overtraining
                best += 1
                if best == 20:
                    if 100*aciertos / batch_size > min_acc:
                        if not os.path.exists('/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time()))+'/'):
                            os.makedirs('/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time()))+'/')
                        saver.save(sess, '/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time()))+'/CNN')
                    print("Training stopped to avoid overtraining")
                    break
            else:
                best = 0
        if 100*aciertos / batch_size > min_acc:
            min_acc = 100*aciertos / batch_size
            if not os.path.exists('/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time()))+'/'):
                os.makedirs('/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time()))+'/')
            saver.save(sess, '/home/pzampella/NimSet/CNN/Hands/Tensorflow/CNN_' + str(int(100 * aciertos / batch_size)) + '_' + str(step) + '_' + str(int(time.time())) + '/CNN')
            print("CNN saved!")
    print("Optimization Finished!")

    # find predictions on val set
    rand2 = np.random.randint(0, len(testing), size=200)
    predict = tf.argmax(prediction, 1)
    pred = predict.eval({xs: testing[rand2], ys: label_testing[rand2], keep_prob: 1.0})

    aciertos = 0.0
    for i in range(0, 200):
        #print(str(pred[i])+"	"+str(np.argmax(label_testing[i])))
        if pred[i] == np.argmax(label_testing[rand2[i]]):
            aciertos += 1.0
    print("Test validation: " + str(100 * aciertos / 200) + "%")