from tensorflow.keras.callbacks import Callback
from tensorflow import sqrt, reduce_sum

class GetIndividualLossesCallback(Callback):

    def on_test_begin(self, logs={}):
        self.individual_losses = []

    def on_test_batch_end(self, batch, logs={}):
        self.individual_losses.append(logs.get('loss'))


def contrastive_loss(y_true, y_pred):
    return sqrt(reduce_sum((y_true - y_pred) ** 2))