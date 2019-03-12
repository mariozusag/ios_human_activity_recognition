# Mario Zusag mariozusag@gmail.com
# 11.03.19
# Purpose:
# Description:
from src.model.keras_base_model import KerasBaseModel


class ConvolutionalNeuralNetwork(KerasBaseModel):

    def __init__(self,
                 batch_size=1000,
                 nb_filter=100,
                 filter_length=5,
                 nb_epoch=5):

        self.batch_size = batch_size
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.nb_epoch = nb_epoch
        self.model = None
        super().__init__('CNN', experiments_path='../../data/output/experimental_results')

    def build_model(self):
        self.model = None

    def train_model(self, X_train, y_train):
        pass
