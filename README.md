# Predicting a Cancer Patient's Survival Using Natural Language Processing with their First Oncologist Document

By John-Jose Nunez, on behalf of the co-authors. 

For the paper: [Predicting the Survival of Patients With Cancer From Their Initial Oncology Consultation Document Using Natural Language Processing](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2801709) published February 27, 2023 in JAMA Network Open. 

Implementation of NLP methods to predict survival using initial oncologist consultations documents.
Most of the data processing and selection was done in a separate repo that I'll keep private for now due to privacy 
concerns, but I'm happy to share code as would be helpful. 

Loosely based on the hedwig NLP repo, but starting fresh due to the large changes
in PyTorch 1.9.0 and especially torchtext 0.10.0

# Training/Fine-tuning NLP Models

Models are ran as python modules. Arguments can be based through the command line. E.g.

python -m cnn --target "surv_mo_60" --batch-size 16

See [models](./models) for the various deployed models. 

See [trainers](./trainers) for the trainers used to train the models

See [evaluators](./evaluators) for the code used to evaluate models

See [results](./results) for the final results and trained models

See [tables](./results) for some code used to generate tables as the raw data used for tables

For training the models on Windows, .bat files are provided that contain the command-line code,
as found in the [bats](./bats) folder. Sorry, I had to use Windows, as IT didn't support Linux on their virtualization.

# Visualizing and Understanding Models

See the [viz](./viz) folder for jupyter notebooks used to visualize and understand the models.

Thank you for your interest in our work! Please don't hesitate to reach out - John-Jose, johnjose.nunez@bccancer.bc.ca

