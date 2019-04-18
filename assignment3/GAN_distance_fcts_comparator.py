import numpy as np
from matplotlib import pyplot as plt
import samplers
from keras.models import Sequential as sq
from keras.layers import Dense
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf
from keras.constraints import max_norm

class GAN_distance_fct_comparator:

    def plot_functions_estimate(self, epochs, distance_fct):
        global graph
        data_points = []

        #Sets the default graph
        graph = tf.get_default_graph()

        #Initialize the model to save defaults weights
        discriminator = sq()
        discriminator.add(Dense(units=64, activation='relu', input_dim=2))
        discriminator.add(Dense(units=64, activation='relu'))
        discriminator.save_weights('model.h5')

        # Get appropriate loss function
        if distance_fct == 'JSD':
            loss_fct = self.JSD_Loss
            discriminator.add(Dense(units=1, activation='sigmoid'))
        if distance_fct == 'Wasserstein':
            loss_fct = self.Wasserstein_Loss
            discriminator.add(Dense(units=1, activation='linear'))
        else:
            assert 'Unknown loss function'

        discriminator.save_weights('model.h5')

        for i in range(21):

            # Reset the model
            K.clear_session()
            tf.reset_default_graph()

            with graph.as_default():
                # Initialize current experiment model
                discriminator = sq()

                # Get appropriate loss function
                if distance_fct == 'JSD':
                    loss_fct = self.JSD_Loss
                    discriminator.add(Dense(units=64, activation='relu', input_dim=2))
                    discriminator.add(Dense(units=64, activation='relu'))
                    discriminator.add(Dense(units=1, activation='sigmoid'))
                if distance_fct == 'Wasserstein':
                    loss_fct = self.Wasserstein_Loss
                    discriminator.add(Dense(units=64, activation='relu', kernel_constraint=max_norm(0.2), input_dim=2))
                    discriminator.add(Dense(units=64, kernel_constraint=max_norm(0.2), activation='relu'))
                    discriminator.add(Dense(units=1,kernel_constraint=max_norm(0.5), activation='linear'))
                else:
                    assert 'Unknown loss function'
                discriminator.load_weights('model.h5')

                if distance_fct == 'Wasserstein':
                    discriminator.compile(loss=loss_fct,
                                              optimizer=SGD(lr=0.1))
                else:
                    discriminator.compile(loss=loss_fct,
                                              optimizer=SGD(lr=0.5))

                phi = round(-1.0 + (0.1 * i), 2)
                for _ in range(epochs):
                    # Create our distributions
                    p_gen = samplers.distribution1(0)
                    q_gen = samplers.distribution1(phi)

                    p = next(p_gen)
                    q = next(q_gen)


                    # Make target dummy for Keras
                    y_dummy = np.zeros(512 * 2)

                    # Train the model on the current distributions

                    discriminator.train_on_batch(np.concatenate((p, q)), y_dummy)

                x = discriminator.get_weights()
                # Create the test distributions
                p_gen = samplers.distribution1(0)
                q_gen = samplers.distribution1(phi)

                p = next(p_gen)
                q = next(q_gen)

                D_x = discriminator.predict(p)
                D_y = discriminator.predict(q)

                if distance_fct == 'JSD':
                    data_points.append(self.JSD(D_x,D_y))
                if distance_fct == 'Wasserstein':
                    data_points.append(self.Wasserstein(D_x,D_y))

        plt.plot(data_points)
        plt.show()

        return 0

    def JSD(self,x,y):

        return np.log(2) + 0.5*np.mean(np.log(x)) + 0.5*np.mean(np.log(1 - y))

    def Wasserstein(self,x,y):
        return np.mean(x) - np.mean(y)

    def JSD_Loss(self,y_true, y_pred):
        D_x = y_pred[:512, :]
        D_y = y_pred[512:, :]

        loss = -(0.5 * K.mean(K.log(D_x)) + 0.5 * K.mean(K.log(1 - D_y)))

        return loss

    def Wasserstein_Loss(self,y_true, y_pred):
        # Sample from the gaussian distribution.
        D_x = y_pred[:512, :]
        D_y = y_pred[512:, :]

        loss = K.mean(D_y) - K.mean(D_x)

        return loss





