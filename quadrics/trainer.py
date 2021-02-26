from quadrics.PointsDataset import PointsDataset
from quadrics.HSQuadricsModel import HSQuadricsModel


class Trainer:

    def __init__(self,
                 model,
                 data,
                 val_data=None,
                 batch_size=256,
                 learning_rate=0.1,
                 save_path=None,
                 log_path=None,
                 start_epoch=1,
                 ):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.log_path = log_path
        self.start_epoch = start_epoch
