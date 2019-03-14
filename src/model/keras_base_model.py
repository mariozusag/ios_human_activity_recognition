# Mario Zusag mariozusag@gmail.com
# 12.03.19
# Purpose:      Framework for testing ML classifiers
# Description:  Initializes a classifier with all necessary parameters and saves the test results in a
#               new folder, which is named after the time of the test run
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.utilities import plot_classification_report
import datetime


class KerasBaseModel:

    def __init__(self, name: str, experiments_path='../../data/output/experimental_results', classifier=None):
        """
        Initializes the classifier framework

        Parameters
        ----------
        name: string
            A descriptive name of the classifier used, like CNN
        experiments_path: string
            The path where the model is saved to
        classifier: sklearn classifier
            A classifier with methods fit(), predict()

        """
        self.name = name
        self.experiments_path = os.path.join(experiments_path,
                                             self.name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

        if not os.path.exists(self.experiments_path):
            print("creating log directory under: " + str(self.experiments_path))
            os.makedirs(self.experiments_path)

        self.classifier = classifier

    def set_classifier(self, classifier):
        """
        Sets the classifier
        :return: None
        """

        self.classifier = classifier

    def train_model(self, X_train: np.array, y_train: np.array):
        """
        Trains the model

        Parameters
        ----------
        X_train: numpy ndarray
            The train data
        y_train: numpy ndarray
            The train labels

        Returns
        -------
        numpy array
        classifier
            A trained classifier
        """

        print('Data dimensions: \n')
        print("self.X_train.shape " + str(X_train.shape))
        print("self.y_train.shape " + str(y_train.shape))
        print('\nTraining model ... \n')
        self.classifier.fit(X_train, y_train)
        return self.classifier

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

        print("Testing model ... ")
        if target_names is None:
            target_names = ['Walking', 'Jogging']
        prediction = self.classifier.predict(X_test).argmax(axis=-1)
        y_test = y_test.argmax(axis=-1)
        plt.figure()
        mat = confusion_matrix(y_test, prediction)
        sns.heatmap(mat, square=True, annot=True, fmt='d', cmap="YlGnBu",
                    cbar=True, xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('true label'),
        plt.xlabel('predicted label')
        plt.savefig(os.path.join(self.experiments_path, "confusion_matrix_gt.pdf"))
        print("confusion matrix: ")
        print(mat)

        plt.figure()
        mat_normalized = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]

        sns.heatmap(mat_normalized, fmt=".2f", square=True, annot=True, cmap="YlGnBu",
                    cbar=True, xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('true label'),
        plt.xlabel('predicted label')
        plt.savefig(os.path.join(self.experiments_path, "confusion_matrix_normalized_gt.pdf"))
        print("confusion matrix normalized: ")
        print(mat_normalized)

        report_dict = classification_report(y_test,
                                            prediction,
                                            output_dict=True,
                                            target_names=target_names,
                                            digits=3)

        report = classification_report(y_test,
                                       prediction,
                                       output_dict=False,
                                       target_names=target_names,
                                       digits=3)
        print(report)
        print("Overall accuracy = {:.2f}%".format(100*(np.sum(y_test == prediction)/len(y_test))))

        json.dump(report_dict,
                  open(os.path.join(self.experiments_path, 'report_dict.json'), 'w'),
                  indent=4,
                  sort_keys=True)

        text_file = open(os.path.join(self.experiments_path, 'report.txt'), 'w')
        text_file.write(report)
        text_file.close()

        plot_classification_report(report, save_as=os.path.join(self.experiments_path, 'report.pdf'))

        json.dump(report,
                  open(os.path.join(self.experiments_path, 'report.json'), 'w'))

        print("Saved all results to {}".format(self.experiments_path))

    @staticmethod
    def plot_history(history, file_path):
        """
        Plots the training history (train/val accuracy, train/val loss) of a classifier

        Parameters
        ----------
        history: keras.model.history
            History object from Keras
        file_path: str
            The file path, where we want to save the history plot to

        Returns
        -------
        None

        """

        print("Plotting training history ... ")
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        # As loss always exists
        epochs = range(1, len(history.history[loss_list[0]]) + 1)

        # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training loss (' + str(str(format(history.history[l][-1], '.2f')) + ')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation loss (' + str(str(format(history.history[l][-1], '.2f')) + ')'))

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(file_path, "loss_plot.pdf"))

        # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training accuracy (' + str(format(history.history[l][-1], '.2f')) + ')')
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation accuracy (' + str(format(history.history[l][-1], '.2f')) + ')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(file_path, "accuracy_plot.pdf"))
