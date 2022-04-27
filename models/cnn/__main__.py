from trainers.cnn_trainer import CNNTrainer
from models.cnn.model import CNN
from models.neural_main import neural_main
from models.cnn.args import get_args

if __name__ == '__main__':
    neural_main("CNN", CNN, CNNTrainer, get_args())
