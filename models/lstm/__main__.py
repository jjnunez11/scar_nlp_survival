from trainers.lstm_trainer import LSTMTrainer
from models.lstm.model import LSTM
from models.neural_main import neural_main
from models.lstm.args import get_args

if __name__ == '__main__':
    neural_main("LSTM", LSTM, LSTMTrainer, get_args())
