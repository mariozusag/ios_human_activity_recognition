# Mario Zusag mariozusag@gmail.com
# 11.03.19
# Purpose:
# Description:
from src.model.keras_base_model import KerasBaseModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, Reshape, MaxPooling1D, Flatten, GlobalMaxPooling1D
import numpy as np
import os
from keras.utils.vis_utils import plot_model


class ConvolutionalNeuralNetwork(KerasBaseModel):

    def __init__(self,
                 batch_size=100,
                 nb_filters=100,
                 filter_size=3,
                 nb_dense_units=100,
                 save_under='../../data/output/experimental_results',
                 nb_epoch=5):
        """
        Initializes a Convolutional Neural Network with the specified parameters

        Parameters
        ----------
        batch_size: int
            Batch size for training
        nb_filters: int
            Number of hidden units in the recurrent layer
        filter_size: int
            Convolution window
        nb_dense_units: int
            Number of hidden units in the dense layer
        save_under: str
            Path to where the model performance should be saved to
        nb_epoch: int
            Number of epochs
        """

        self.batch_size = batch_size
        self.nb_filters = nb_filters
        self.filter_size = filter_size
        self.nb_dense_units = nb_dense_units
        self.nb_epoch = nb_epoch
        self.save_under = save_under
        self.model = None
        super().__init__('CNN', experiments_path=save_under)

    def build_model(self, X_train, y_train):
        """
        Builds a CNN classifier
        Parameters
        ----------
        X_train: numpy ndarray
            Train data in the format [n_samples, n_datapoints, n_features]
        y_train
            Train labels in the format [n_samples, n_labels]
        Returns
        -------
        keras Sequential model

        """
        n_labels = y_train.shape[1]
        input_shape = X_train.shape[1]
        n_datapoints = int(input_shape / 3)

        self.model = Sequential()
        self.model.add(Reshape((n_datapoints, 3), input_shape=(input_shape,)))
        self.model.add(Convolution1D(filters=self.nb_filters,
                                     kernel_size=self.filter_size,
                                     activation='relu',
                                     input_shape=(n_datapoints, 3)))
        self.model.add(Convolution1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.25))
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

    def test_model(self, X_test: np.array, y_test: np.array, target_names=None):
        """
        Tests the trained classifier on X_test and creates several performance reports

        Parameters
        ----------
        X_test: np.array
        y_test: np.array
        target_names: list
            A list of strings with the targets

        Returns
        -------
        None

        """
        super(ConvolutionalNeuralNetwork, self).test_model(X_test, y_test, target_names=target_names)
        self.model.save(filepath=os.path.join(self.experiments_path, "model.h5"))


if __name__ == '__main__':
    cnn = ConvolutionalNeuralNetwork()
    data = np.load("../../data/train_test/motion_sense_data.npz")
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    model = cnn.build_model(X_train=X_train_cnn, y_train=y_train_cnn)
    cnn.test_model(X_test=X_test_cnn, y_test=y_test_cnn)