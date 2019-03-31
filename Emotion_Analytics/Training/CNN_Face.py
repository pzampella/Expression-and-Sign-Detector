from __future__ import print_function
import cv2
import random
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from sklearn.decomposition import PCA
#from sklearn import preprocessing
import os
from scipy import signal
import dlib
from PIL import Image
from skimage import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(27)

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def contraste(imagen):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    contraste = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return contraste

# Import data
path_imagen = "/home/pzampella/NimSet/CropWhiteBackground/"
path_archivo = "/home/pzampella/NimSet/NimStim_ratings.csv"
data = np.genfromtxt(path_archivo, dtype=None, delimiter=',')
#random.shuffle(data)
# Parameters
learning_rate = 0.0001
training_iters = 3000
new_dim = 32
pct_test = 0.1
conv = 5
batch_size = 16
dropout = 0.75 # Dropout, probability to keep units
# print(path_imagen + data[30][0].upper())
#pca = PCA(n_components=new_dim, copy=False)
archivos = []
etiquetas = []
nombres = []
for i in range(0, len(data)):
    if os.path.exists(path_imagen + data[i][0].upper()):
        nombres.append(data[i][0].upper())
        aux0 = cv2.imread(path_imagen + data[i][0].upper(), True)
        aux1 = cv2.cvtColor(contraste(aux0), cv2.COLOR_BGR2GRAY)
        detected_faces = detect_faces(aux1)
        if len(detected_faces) > 0:
            face = np.array(Image.fromarray(aux1).crop(detected_faces[0]))
            aux2 = cv2.resize(face, (new_dim, new_dim))
            # plt.imshow(aux2, cmap="gray")
            # plt.show()
            #aux3 = cv2.Canny(aux2, 150, 200)
            # sp_noise = np.random.randint(0, 255, (new_dim, new_dim), dtype=int)
            # for k in range(0, new_dim):
            #     for j in range(0, new_dim):
            #         if sp_noise[j][k] > 250:
            #             aux2[j][k] = 255
            #         if sp_noise[j][k] < 5:
            #             aux2[j][k] = 0
            #aux3 = sp_noise()
            # plot = plt.imshow(aux2, cmap="gray")
            # plt.show()
            # plot = plt.imshow(aux3, cmap="gray")
            # plt.show()
            aux = np.concatenate(aux2)
            #aux = cv2.imread(path_imagen + data[i][0].upper(), False)
            archivos.append(aux)
            numero = data[i][1]
            if str(numero) == "1":
                etiquetas.append([1, 0, 0, 0, 0, 0, 0])
            if str(numero) == "2":
                etiquetas.append([0, 1, 0, 0, 0, 0, 0])
            if str(numero) == "3":
                etiquetas.append([0, 0, 1, 0, 0, 0, 0])
            if str(numero) == "4":
                etiquetas.append([0, 0, 0, 1, 0, 0, 0])
            if str(numero) == "5":
                etiquetas.append([0, 0, 0, 0, 1, 0, 0])
            if str(numero) == "6":
                etiquetas.append([0, 0, 0, 0, 0, 1, 0])
            if str(numero) == "7":
                etiquetas.append([0, 0, 0, 0, 0, 0, 1])
#projected = pca.fit_transform(preprocessing.scale(np.array(archivos).astype(np.float64)))
#test = int(math.floor(len(archivos)*pct_test))
# test = 55
# testing = np.array(archivos[0:test])
# training = np.array(archivos[test:len(archivos)])
# label_testing = np.array(etiquetas[0:test]).astype(np.float64)
# label_training = np.array(etiquetas[test:len(etiquetas)]).astype(np.float64)
rand = random.sample(range(10, 43), 4)
print(rand)
testing2 = []
training2 = []
label_testing2 = []
label_training2 = []
for i in range(0, len(archivos)):
    #if nombres[i][0:2] == "27" or nombres[i][0:2] == "37" or nombres[i][0:2] == "33" or nombres[i][0:2] == "22":
    if nombres[i][0:2] == str(rand[0]) or nombres[i][0:2] == str(rand[1]) or nombres[i][0:2] == str(rand[2]) or nombres[i][0:2] == str(rand[3]):
        testing2.append(archivos[i])
        label_testing2.append(etiquetas[i])
    else:
        training2.append(archivos[i])
        label_training2.append(etiquetas[i])
test = len(label_testing2)
testing = np.array(testing2)
training = np.array(training2)
label_testing = np.array(label_testing2)
label_training = np.array(label_training2)
# Network Parameters
n_input = new_dim*new_dim # data input (img shape: 640*490)
n_classes = 7 # total classes (7 emotions: 0=anger, 1=neutral, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
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
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_input])   # 28x28
ys = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, new_dim, new_dim, 1])/255.

## conv1 layer ##
W_conv1 = weight_variable([conv, conv, 1, 32], "W_conv1") # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], "b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([conv, conv, 32, 64], "W_conv2") # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
if new_dim==10:
    W_fc1 = weight_variable([144*4, 1024], "W_fc1")  #10x10
if new_dim == 20:
    W_fc1 = weight_variable([400*4, 1024], "W_fc1")  #20x20
#W_fc1 = weight_variable([576, 1024], "W_fc1")  #650x506
if new_dim == 100:
    W_fc1 = weight_variable([40000, 1024], "W_fc1")  #100x100
if new_dim == 32:
    W_fc1 = weight_variable([4096, 1024], "W_fc1")     #32x32
if new_dim == 64:
    W_fc1 = weight_variable([16384, 1024], "W_fc1")     #64x64
b_fc1 = bias_variable([1024], "b_fc1")
if new_dim==10:
    h_pool2_flat = tf.reshape(h_pool2, [-1, 144*4]) #10x10
if new_dim == 20:
    h_pool2_flat = tf.reshape(h_pool2, [-1, 400*4]) #20x20
#h_pool2_flat = tf.reshape(h_pool2, [-1, 576]) #650x506
if new_dim == 100:
    h_pool2_flat = tf.reshape(h_pool2, [-1, 40000]) #100x100
if new_dim == 32:
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4096]) #32x32
if new_dim == 64:
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16384]) #64x64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, n_classes], "W_fc2")
b_fc2 = bias_variable([n_classes], "b_fc2")
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
    min_acc = 80.0
    # Keep training until reach max iterations
    while step < training_iters:
        rand = np.random.randint(0, len(training), size=batch_size)
        sess.run(train_step, feed_dict={xs: training[rand], ys: label_training[rand], keep_prob: dropout})
        step += 1
        if step % 10 == 0:
            acc = compute_accuracy(training[rand], label_training[rand])
            if step % 10 == 0:
                aciertos = 0.0
                rand2 = np.random.randint(0, test, size=1)
                #plot = plt.imshow(np.reshape(training[rand[0:1]], (100, 100)))
                #plt.show()
                # print(training[rand[0:1]])
                # print(prediction.eval({xs: training[rand[0:1]], ys: label_training[rand[0:1]], keep_prob: 1.0}))
                # print(label_training[rand[0:1]])
                # print("-----")
                # print(testing[[rand2]])
                # print(prediction.eval({xs: testing[[rand2]], ys: label_testing[[rand2]], keep_prob: 1.0}))
                # print(label_testing[[rand2]])
                #plot = plt.imshow(np.reshape(testing[[0]], (100, 100)))
                #plt.show()
                predict = tf.argmax(prediction, 1)
                pred = predict.eval({xs: testing, ys: label_testing, keep_prob: 1.0})
                for i in range(0, test):
                    if pred[i] == np.argmax(label_testing[i]):
                        aciertos += 1.0
                    # if pred[i] == 0:
                    #     print("Neutral")
                    # if pred[i] == 1:
                    #     print("Anger")
                    # if pred[i] == 2:
                    #     print("Contempt")
                    # if pred[i] == 3:
                    #     print("Disgust")
                    # if pred[i] == 4:
                    #     print("Fear")
                    # if pred[i] == 5:
                    #     print("Happy")
                    # if pred[i] == 6:
                    #     print("Sad")
                    # if pred[i] == 7:
                    #     print("Surprise")
                print("Iteration: " + str(step) + r"/" + str(training_iters) + "   Training Accuracy: " + str(acc)+"    Test Accuracy: " + str(100 * aciertos / test) + "%")
            else:
                print("Iteration: " + str(step) + r"/" + str(training_iters) + "   Training Accuracy: " + str(acc))
            if acc > 0.95: #Early stop to avoid overtraining
                best += 1
                if best == 10:
                    if 100*aciertos / test > min_acc:
                        if not os.path.exists('/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time()))+'/'):
                            os.makedirs('/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time()))+'/')
                        saver.save(sess, '/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time()))+'/CNN')
                    print("Training stopped to avoid overtraining")
                    break
            else:
                best = 0
        if 100*aciertos / test > min_acc:
            min_acc = 100*aciertos / test
            print(min_acc)
            if not os.path.exists('/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time()))+'/'):
                os.makedirs('/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time()))+'/')
            saver.save(sess, '/home/pzampella/NimSet/CNN/Face/Tensorflow/CNN_' + str(int(100 * aciertos / test)) + '_' + str(step) + '_' + str(int(time.time())) + '/CNN')
            print("CNN saved!")
    print("Optimization Finished!")

    # find predictions on val set
    predict = tf.argmax(prediction, 1)
    pred = predict.eval({xs: testing, ys: label_testing,  keep_prob: 1.0})

    aciertos = 0.0
    for i in range(0, test):
        print(str(pred[i])+"	"+str(np.argmax(label_testing[i])))
        if pred[i] == np.argmax(label_testing[i]):
            aciertos += 1.0
        # if pred[i] == 0:
        #     print("Neutral")
        # if pred[i] == 1:
        #     print("Anger")
        # if pred[i] == 2:
        #     print("Contempt")
        # if pred[i] == 3:
        #     print("Disgust")
        # if pred[i] == 4:
        #     print("Fear")
        # if pred[i] == 5:
        #     print("Happy")
        # if pred[i] == 6:
        #     print("Sad")
        # if pred[i] == 7:
        #     print("Surprise")
    print("Test validation: " + str(100 * aciertos / test) + "%")