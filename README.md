# Running Instructions 

To train the base QG-Net model, run ```python qg_net_base.py```
To train the BERT based QG-Net model, run ```python qg_net_bert.py```

An example of running inference to compute the metrics has been shown in ```inference.py```

# Data Link 
The dataset used along with the trained weights can be found at - https://drive.google.com/drive/folders/1Jlo4wUBKDapEhR6D2-uaR3N8cctCy8lF?usp=sharing


# QG-Net Implementation

``` RNNEncoder```: This class encode the input using a bidirectional input and produces representations for each word in the context  

```RNNDecoder```: This class decode the output of Encoder in sequential manner with attention and pointer network (used for calculating probabilities from source side).   
```QGNet```: This class combine the encoder and decoder. the forward pass uses teacher forcing if a target is available. else special token '<sos>' is used as input for the first decoder step.  

# QG-Net with Bert 
```QADataset```: Dataset used for converting data to form usable by Huggingface transformers   

```BertEncoder```: We used pre-trained encoder from bert 'bert-base-uncased' which assume capital and small alphabets as same.  

```BertDecoder```: The decoder will decode the encoder output and calculate probability distribution using attention and pointer network. Uses a BertLMHeadModel for decoding.  

```BertPointerNetwork```-:  This class combine the encoder and decoder. the forward pass uses teacher forcing if a target is available. else special token '[CLS]' is used as input for the first decoder step.  

# Util 
## Functions
```train_step```: this function will check our model on train data and return how much loss we obtain for this data.    
```eval_step```:this function will evalute our model on test data and test how much loss we obtain for this data.   
```train_model```:The function will train our model for certain epochs which are 10 in our case.also this function is used to print losses and to save model at `pt` format.