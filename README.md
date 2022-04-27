# Prediction of Cancer Needs Using NLP with Physician Documents

Fresh implementation of NLP methods to predict cancer needs using physician documents. 

Loosely based on the hedwig NLP repo, but starting fresh due to the large changes
in  PyTorch 1.9.0 and especially torchtext 0.10.0

# Training/Fine-tuning NLP Models

Models are ran as python modules. Arguments can be based through the command line. E.g.

python -m cnn --target "need_emots_1" --batch-size 16

See [models](./models) for the various deployed models. 

See [trainers](./trainers) for the trainers used to train the models

See [evaluators](./evaluators) for the code used to evaluate models

For training the models on windows, .bat files are provided that contain the command-line code,
as found in the [bats](./bats) folder. 

# Visualizing and Understanding Models

See the [viz](./viz) folder for jupyter notebooks used to visualize and understand how are 
models are trained and fine-tuned. 

