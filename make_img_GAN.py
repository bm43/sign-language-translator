import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

###########################load data#########################3
dir1 = "C:/Users/SamSung/Desktop/uni/y2/EDP/Angry"
dir2="C:Users/SamSung/Desktop/uni/y2/EDP/Hungry"
dir3="C:Users/SamSung/Desktop/uni/y2/EDP/What"
#생성하고 싶은 이미지들마다 다른 directory
labels = ["Angry", "Hungry", "What"]

train=[]
y=[]
tdata=[]


def cr_tdata():
    path = dir1
    class_num = labels.index(labels[0])#change for each folder
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(108,108))
            tdata.append([new_array, class_num])
        except:
            pass

cr_tdata()
random.shuffle(tdata)

for features, label in tdata:
    train.append(features)
    y.append(label)
train=np.asarray(train, dtype=np.int32)
train=train.reshape(-1,108*108)
train=np.divide(train,255)
#print(len(X))
#print(len(X[0]))
##############################################################



###################model################3
total_epochs=100
batch_size = 100
learning_rate = 0.0002

def generator(z,reuse=False):
    if reuse==False:
        with tf.variable_scope(name_or_scope = "Gen") as scope :
            gw1 = tf.get_variable(name = "w1",
                                  shape = [128,256],
                                  initializer= tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb1 = tf.get_variable(name = "b1",
                                 shape = [256],
                                 initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gw2 = tf.get_variable(name = "w2",
                                  shape = [256, 108*108],
                                  initializer= tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb2 = tf.get_variable(name = "b2",
                                 shape = [108*108],
                                 initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
    hidden = tf.nn.relu( tf.matmul(z , gw1) + gb1 )
    output = tf.nn.sigmoid( tf.matmul(hidden, gw2) + gb2 )

    return output#fake image


def discriminator(x,reuse=False):
    if reuse == False:
        with tf.variable_scope(name_or_scope="Dis") as scope :

            dw1 = tf.get_variable(name = "w1",
                                  shape = [108*108, 256],
                                  initializer = tf.random_normal_initializer(0.0, 0.01) )
            db1 = tf.get_variable(name = "b1",
                                  shape = [256],
                                  initializer = tf.random_normal_initializer(0.0, 0.01) )
            dw2 = tf.get_variable(name = "w2",
                                  shape = [256, 1],
                                  initializer = tf.random_normal_initializer(0.0, 0.01) )
            db2 = tf.get_variable(name = "b2",
                                  shape = [1],
                                  initializer = tf.random_normal_initializer(0.0, 0.01) )
    hidden = tf.nn.relu( tf.matmul(x , dw1) + db1 )  #[-, 256]
    output = tf.nn.sigmoid( tf.matmul(hidden, dw2)  + db2 )  #real (1) or not real(0) [-, 1]

    return output

def random_noise(batch_size) :
    return np.random.normal(size=[batch_size , 128])

g=tf.Graph()
with g.as_default():
    #input
    X=tf.placeholder(tf.float32, [None, 108*108])
    Z=tf.placeholder(tf.float32, [None, 128])

    #using the nets
    fake_x = generator(Z)
    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X , True)

    #losses
    #both high are good
    g_loss = tf.reduce_mean( tf.log(result_of_fake) )
    d_loss = tf.reduce_mean( tf.log(result_of_real) + tf.log(1 - result_of_fake) )

    #train the nets to max losses
    t_vars = tf.trainable_variables()

    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)

    g_train = optimizer.minimize(-g_loss, var_list= g_vars)
    d_train = optimizer.minimize(-d_loss, var_list = d_vars)

#training the neural nets
with tf.Session(graph = g) as sess :
    sess.run(tf.global_variables_initializer())

    total_batchs = int(train.shape[0] / batch_size)

    for epoch in range(total_epochs) :

        for batch in range(total_batchs) :
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]  # [batch_size , 784]
            batch_y = train_y[batch * batch_size : (batch+1) * batch_size]  # [batch_size,]
            noise = random_noise(batch_size)  # [batch_size, 128]

            sess.run(g_train , feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : batch_x , Z : noise})

        #check learning performance every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 1  :
            print("=======Epoch : ", epoch , " =======================================")
            print("gen loss : " ,gl )
            print("disc loss : " ,dl )
            print("training...")


        #imshow results every 20

        if epoch == 0 or (epoch + 1) % 20 == 0  :
            sample_noise = random_noise(1)

            generated = sess.run(fake_x , feed_dict = { Z : sample_noise})

            fig, ax = plt.subplots(1, 1, figsize=(1, 1))

            ax[0].set_axis_off()
            ax[0].imshow( np.reshape( generated[0], (108, 108)) )

            plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)


    print('done.')
