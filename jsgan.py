import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as L
from functools import partial
import numpy as np
import pandas as pd


def get_tf_dataset(dataset, batch_size):
    N_EXAMPLES = len(dataset)
    shuffler = tf.contrib.data.shuffle_and_repeat(N_EXAMPLES)
    dataset_tf = tf.data.Dataset.from_tensor_slices(dataset)
    suffled_ds = shuffler(dataset_tf)
    
    dataset_final = suffled_ds.batch(batch_size).prefetch(1)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset_final)
    return iterator.get_next()

def sample_noise_batch(bsize, x_dim):
    return np.random.normal(size=(bsize, x_dim)).astype('float32')


class Model:
    def fit(self, X, Y):
        sess = tf.InteractiveSession()
        
        TOTAL_ITERATIONS = int(5e3)
        DISCRIMINATOR_ITERATIONS = 5
        x_dim, y_dim = 3, 5
        train_batch_size = 1000
        disc_lr = 1e-3

        gen_activation = tf.keras.activations.elu
        with tf.name_scope("Generator"):
            generator = Sequential(name="Generator")
            generator.add(L.InputLayer([x_dim], name='noise'))
            for i in range(5):
                generator.add(L.Dense(128, activation=gen_activation))
            generator.add(L.Dense(y_dim))

            
        disc_activation = partial(tf.keras.activations.relu, alpha=0.3)
        with tf.name_scope("Discriminator"):
            discriminator = Sequential(name="Discriminator")
            discriminator.add(L.InputLayer([y_dim]))
            for i in range(5):
                discriminator.add(L.Dense(128, activation=disc_activation))         
            discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))

        
        real_data = get_tf_dataset(Y, train_batch_size)
        real_data = tf.dtypes.cast(real_data, tf.float32)
        noise_batch_size = tf.placeholder(tf.int32, shape=[], name="noise_batch_size")
        noise = tf.random_normal([noise_batch_size, x_dim], dtype=tf.float32, name="noise")
        
        
        generated_data = generator(noise)
        logp_real = discriminator(real_data)
        logp_gen = discriminator(generated_data)
        
        
        
        disc_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])
        disc_optimizer = tf.train.GradientDescentOptimizer(disc_lr).minimize(
                disc_loss, var_list=discriminator.trainable_weights)
        
        gen_loss = -tf.reduce_mean(logp_gen[:,1])
        tf_iter = tf.Variable(initial_value=0, dtype=tf.int32)
        gen_lr = tf.train.exponential_decay(5e-4, tf_iter, 150, 0.98)
        gen_optimizer = tf.group(
            tf.train.AdamOptimizer(gen_lr).minimize(
                gen_loss, var_list=generator.trainable_weights),
            tf.assign_add(tf_iter, 1))
        
        learning_summary = tf.summary.merge([
            tf.summary.scalar("disc_loss", disc_loss),
            tf.summary.scalar("gen_loss", gen_loss),
        ])

        sess.run(tf.global_variables_initializer())

        for epoch in range(TOTAL_ITERATIONS):
            for i in range(DISCRIMINATOR_ITERATIONS):
                sess.run(disc_optimizer, {noise_batch_size: train_batch_size})
            summary, _, _ = sess.run(
                [learning_summary, gen_optimizer, tf_iter], 
                {noise_batch_size: train_batch_size})
    
        self.generator = generator
        self.y_cols = Y.columns
        self.x_dim = x_dim
        self.sess = sess
            
    def predict(self, X):
        Y_pred = self.generator.predict(
            sample_noise_batch(bsize=len(X), x_dim=self.x_dim))
        return pd.DataFrame(Y_pred, columns=self.y_cols)
