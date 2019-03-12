# Mario Zusag mariozusag@gmail.com
# 11.03.19
# Purpose:
# Description:

from src.model.keras_base_model import KerasBaseModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Reshape, Bidirectional
import numpy as np
from keras.utils import to_categorical
import os
from keras.utils.vis_utils import plot_model


class RecurrentNeuralNetwork(KerasBaseModel):

    def __init__(self,
                 batch_size=100,
                 nb_units=100,
                 nb_dense_units=100,
                 save_under='../../data/output/experimental_results',
                 nb_epoch=5):
        """
        Initializes a Recurrent Neural Network with the specified parameters

        Parameters
        ----------
        batch_size: int
            Batch size for training
        nb_units: int
            Number of hidden units in the recurrent layer
        nb_dense_units: int
            Number of hidden units in the dense layer
        save_under: str
            Path to where the model performance should be saved to
        nb_epoch: int
            Number of epochs
        """

        self.batch_size = batch_size
        self.nb_units = nb_units
        self.nb_dense_units = nb_dense_units
        self.nb_epoch = nb_epoch
        self.save_under = save_under
        self.model = None
        super().__init__('LSTM', experiments_path=save_under)

    def build_model(self, X_train, y_train):
        """
        Builds an RNN classifier
        Parameters
        ----------
        X_train: numpy ndarray
            Train data in the format [n_samples, n_datapoints, n_features]
        y_train
            Train labels in the format [n_samples, ]
        Returns
        -------
        keras Sequential model

        """
        n_labels = y_train.shape[1]
        input_shape = X_train.shape[1]
        n_datapoints = int(input_shape/3)

        self.model = Sequential()
        self.model.add(Reshape((n_datapoints, 3), input_shape=(input_shape,)))
        self.model.add(Bidirectional(LSTM(self.nb_units, input_shape=(None, n_datapoints))))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.nb_dense_units, activation='relu'))
        self.model.add(Dense(n_labels, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        plot_model(self.model, to_file=os.path.join(self.experiments_path, 'model_plot.png'),
                   show_shapes=True,
                   show_layer_names=True)

        print(self.model.summary())

        self.model.fit(X_train, y_train, epochs=self.nb_epoch, batch_size=self.batch_size, verbose=1)
        super().set_classifier(self.model)

        return self.model

    def test_model(self, X_test, y_test, target_names=None):
        """
        Runs several performance measures and saves the results
        Parameters
        ----------
        X_test: numpy ndarray
        y_test
        target_names

        Returns
        -------

        """
        super(RecurrentNeuralNetwork, self).test_model(X_test, y_test, target_names=target_names)
        self.model.save(filepath=os.path.join(self.experiments_path, "model.h5"))


if __name__ == '__main__':
    rnn = RecurrentNeuralNetwork()
    data = np.load("../../data/train_test/motion_sense_data.npz")
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model = rnn.build_model(X_train=X_train, y_train=y_train)
    rnn.test_model(X_test=X_test, y_test=y_test)
