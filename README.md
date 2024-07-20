Some of the code for this project was referenced from: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/README.md


# Project Report
## By Nelson Goosman

### Introduction (Statement of data mining task)
For this project, I wanted to create a sentiment analysis tool that can take a sentence as input and classify it as either positive or negative. To do this, I trained my models on an IMDB dataset consisting of 25,000 movie reviews, using 75% of that data for training and the remaining 25% for testing and validation. The models I chose to use are a Neural Bag of Words (NBoW) and a Long Short Term Memory Model (LSTM). I then saved these models to files, where I can load them into a repl like tool on the command line. The user can enter the type of model they want to use as well as the sentence they want to evaluate, and the model will perform inference on that text and generate a score of either positive or negative as well as a confidence score (probability) of the answer. Each model was trained for 10 Epochs.

### Methodology / Technical Approach
Both models begin by loading in the imbd dataset
from the pytoch datasets module. As stated above, this contains 25000 labeled movie reviews. They also use the torchtext "basic english" tokenizer. The tokenizer exists to take a senetence and convert it to a series of integers. For example, the sentence "This is a great movie" could get converted to the tokens "4, 15, 34, 2, 86". The dataset is split into training, testing, and validation data. This is to ensure that the model does not overfit on the testing data and prevents it from generalizing poorly to other data. After the dataset is loaded, it is converted to tensors, which is a multidimensional array like data structure that pytorch can perform efficent matrix operations on. Since the model expects inputs of a certain size, I set the embedding_dimension to 300 tokens, so longer sentences are trimmed to length 300 and shorter sentences are padded with a padding token to be length 300. Words that appear less than 5 times in the dataset are discarded. Additionally, I set the batch size to 512 meaning that the model looks at 512 examples in parallel before updating its weights. This makes the model train faster. Finally, both models used the adam optimizer and were trained for 10 epochs. They both perform a binary classification of either positive (1) or negative (0) as well as a confidence score from 0 - 1.0. The models both support running on a GPU to speed up training but my laptop does not support this (more on this later). I began by training a NBoW model as it is the simplest architexture to accomplish this task. It starts by encoding the words as integers and then pools the words, meaning it calculates the average of all the words in the data it is training on. The result of this is fed through a linear classification layer to produce a classification. I then implimented a LSTM, which is a more technically complicated model. Like the NBoW model, the text is first embedded into a series of vectors. The embedding is then fed into two LSTM layers. Each layer of the lstm contains cells which maintain an internal state. This state is updated at each step. The lstm layers are bidirectional, meaning they process the input both foreward and backwards. This allows it to capture context from the sentence in both directions. Finally, the LSTM has a dropout rate of .5, meaning that the cells in each layer will have a 50% probability to turn off randomly each epoch. This helps protect against overfitting.

NBOW Architexture:  \
       +-----------+ \
       |  postive  | \
       +-----------+ \
            ^ \
            | \
       +-----------+ \
       |   Linear  | \
       |    Layer  | \
       +-----------+ \
            ^ \
            | \
       +-----------+ \
       |  Pooling  | \
       +-----------+ \
            ^ \
            | \
  +-----------+   +-----------+   +-----------+   +-----------+   +-----------+ \
  |Embedding  |   |Embedding  |   |Embedding  |   |Embedding  |   |Embedding  | \
  |  Layer    |   |  Layer    |   |  Layer    |   |  Layer    |   |  Layer    | \
  +-----------+   +-----------+   +-----------+   +-----------+   +-----------+ \
       ^                ^                ^                ^                ^ \
       |                |                |                |                | \
      "I"            "enjoyed"         "this"           "movie"            "!" \

LSTM Architexture: \
         +-----------------------+ \
         |      Sentiment        | \
         |     Classification    | \
         +-----------------------+ \
                  ^ \
                  | \
         +-----------------------+ \
         |       Linear Layer    | \
         +-----------------------+ \
                  ^ \
                  | \
         +-----------------------+ \
         |     Concatenation     | \
         +-----------------------+ \
                  ^ \
                  | \
  +-----------------+     +-----------------+ \
  | Forward LSTM    |     | Backward LSTM   | \
  | Layer 2 Output  |     | Layer 2 Output  | \
  +-----------------+     +-----------------+ \
         ^                          ^ \
         |                          | \
  +-----------------+     +-----------------+ \
  | Forward LSTM    |     | Backward LSTM   | \
  | Layer 1 Output  |     | Layer 1 Output  | \
  +-----------------+     +-----------------+ \
         ^                          ^ \
         |                          | \
         |                          | \
  +---------------------------------------------------+ \
  |              Word Embeddings/Input                | \
  +---------------------------------------------------+ \
         |        |        |         |        | \
        "I"    "enjoyed" "this"   "movie"    "!" \

### Evaluation of Methodology
The challenge of this project was that the dataset only contained positive or negative classification, so neutral text cannot be classified. Additionally, all of the data was about movies, so the model is more profficent at classifying movie reviews. This can cause it to misclassify text not related to movies. Additionally, my laptop does not support CUDA so I cannot train it on a GPU. Training on the CPU for 10 epochs took around 12 hours so I was only able to train it once and did not have time to tweak the hyperparameters.
### Results
After 10 Epochs of training, the models performed as follows:
-----------------------LSTM-----------------------
train_loss: 0.207, training accuracy: 0.920
validation loss: 0.295, validation accuracy: 0.888
-----------------------NBOW-----------------------
training loss: 0.293, training accuracy: 0.900
validation loss: 0.351, validation accuracy: 0.862
### Instructions
to run this model, install the requirements.txt file using either pip, conda, 
or mamba. Then run util.py and follow the instructions on the command line. 




