# Code for this project was referenced from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb
import torch
import pandas as pd
import numpy as np
import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import torchtext.data
import torchtext.vocab
import utils

# the maximum length of a tokens we will give the model at once
MAX_TOKEN_LEN = 256



class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # takes the one hot encoded words and embeds them in a vector space
        embedded = self.embedding(ids)
        # pools embedded layer by finding the mean of the data
        pooled = embedded.mean(dim=1)
        # runs pool through linear layer
        prediction = self.fc(pooled)
        return prediction

def train_model(data_loader, model, criterion, optimizer, device):
    '''
    Trains the model on one batch of data
    '''
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate_model(dataloader, model, criterion, device):
    '''
    Evaluates the accuracy of the model 
    '''
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    '''
    Gets accuracy of the model during training
    '''
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def predict_sentiment(text, model, tokenizer, vocab, device):
    '''
    Runs inference on the model, returns the predicted class and probability
    '''
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability





if __name__ == '__main__':
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    train, test = datasets.load_dataset('imdb', split=['train', 'test'])

    # tokenize train and test data
    train = train.map(
        utils.tokenize_str, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_TOKEN_LEN}
    )
    test = test.map(
        utils.tokenize_str, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_TOKEN_LEN}
    )

    # split testing data up into test and validation data
    train_test_split = train.train_test_split(test_size=0.25)
    train = train_test_split["train"]
    validation = train_test_split["test"]

    # minimim frequency a token must appear in the dataset to be included
    minimum_frequency = 5
    # unk: if an unknown token is encountered, it is converted ot unc
    #pad: padding token used to pad sentences
    special_tokens = ["<unk>", "<pad>"]

    # build model vocabulary
    vocabulary = torchtext.vocab.build_vocab_from_iterator(
        train["tokens"],
        min_freq=minimum_frequency,
        specials=special_tokens,
    )

    unknown = vocabulary["<unk>"]
    pad = vocabulary["<pad>"]
    # if an unknown word is encountered, unk is used
    vocabulary.set_default_index(unknown)
    # save vocabulary to import to other files
    torch.save(vocabulary, 'vocabulary.pth')

    # convert training data from words to one hot encodings
    train = train.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocabulary})
    validation = validation.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocabulary})
    test = test.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocabulary})

    # convert to tensors
    train = train.with_format(type="torch", columns=["ids", "label"])
    validation = validation.with_format(type="torch", columns=["ids", "label"])
    test = test.with_format(type="torch", columns=["ids", "label"])

    BATCH_SIZE = 512

    train_data_loader = utils.get_data_loader(train, BATCH_SIZE, pad, shuffle=True)
    validation_data_loader = utils.get_data_loader(validation, BATCH_SIZE, pad)
    test_data_loader = utils.get_data_loader(test, BATCH_SIZE, pad)

    # set up variables for initializing model class
    vocabulary_size = len(vocabulary)
    embedding_dim = 300
    output_dim = len(train.unique("label"))
    model = NBoW(vocabulary_size, embedding_dim, output_dim, pad)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is running on: {device}")
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 10
    best_valid_loss = float("inf")

    # training loop
    for epoch in range(n_epochs):
        train_loss, train_acc = train_model(
            train_data_loader, model, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate_model(validation_data_loader, model, criterion, device)
        if validation_loss < best_valid_loss:
            best_valid_loss = validation_loss
            torch.save(model, "BagOfWords.pt")
        print(f"epoch: {epoch}")
        print(f"training loss: {train_loss:.3f}, training accuracy: {train_acc:.3f}")
        print(f"validation loss: {validation_loss:.3f}, validation accuracy: {validation_accuracy:.3f}")

    
