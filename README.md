# Kinase Substrate Prediction
This is a repository for Kinase Substrate Prediction. Based on kinase-substrate phosphorylation data in [PhosphoSitePlus](https://www.phosphosite.org/) and corresponding embedding features in [Bioteque](https://bioteque.irbbarcelona.org/), I build a MLP for kinase-substrate prediction.

The model is trained through 5-fold cross validation 10 times. The evaluation metrics are AUROC, precision, recall and accuracy. 

## Usage

### Preparation
Firstly run the following command to download the embedding features and generate samples.You can get the .csv files of the positive and ten negative samples in ```./data/samples```(default), which can be used to train and evaluate.

According to the data of [PhosphoSitePlus] (https://www.phosphosite.org/) (i.e. ./data/Kinase_Substrate_Dataset.csv), the number of positive samples is 7456, which the negatives is the same as it. Note that we need to train the model 10 times, so the number of negative samples is 10 times as much as the positives.

The preprocess.py script has two parameters: ```--data_output``` and ```--repeat```, which represent the output path of samples and the num of negative samples respectively. The default value of ```--data_output``` is ```./data/samples```, and the default value of ```--repeat``` is 10. 



```
python preprocess.py
```

### Training
Then run the following command to train the model. There will be 50 models, i.e. fold_num * repeat_num. The models will be saved in ```./model```(default).

The parameters of train.py are in train_param_parsing.py.


```
python train.py
```

### Result
After training, it will plot a boxplot diagram of the evaluation metrics. The metrics data and diagram will be saved in ```./result```.

### My result
The result of my model is as follows:

<img src="boxplot diagram/boxplot.png"/>


