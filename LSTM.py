import collections
import utils
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


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # converts words to embedded layer
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
       
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)

def train(dataloader, model, criterion, optimizer, device):
    '''
    Trains the model on one batch of data
    '''
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(dataloader, model, criterion, device):
    '''
    Evaluates the accuracy of the model
    '''
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_collate_fn(pad_index):
    '''
    Returns function to pad a vector of words
    '''
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    '''
    Creates a data loader that prepares the data for training the model 
    '''
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


def get_accuracy(prediction, label):
    '''
    Returns accuracy: correct predictions / total predictions
    '''
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def tokenize_str(example, tokenizer, max_length):
    '''
    Converts strings to tokens via one hot encoding. Trims to max length
    '''
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}

def predict_sentiment(text, model, tokenizer, vocab, device):
    '''
    Runs inference on a given string of text and returns predicted class, predicted probability
    '''
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

if __name__ == '__main__':

    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    MAX_TOKEN_LEN = 256

    # tokenize train and test data
    train_data = train_data.map(
        tokenize_str, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_TOKEN_LEN}
    )
    test_data = test_data.map(
        tokenize_str, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_TOKEN_LEN}
    )

    # split into train and validation data
    train_test_data = train_data.train_test_split(test_size=0.25)
    train_data = train_test_data["train"]
    validation_data = train_test_data["test"]
    
    min_freq = 5
    special_tokens = ["<unk>", "<pad>"]
    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)

    # maps data from tokens to ints via one hot encoding, then convert to tensors
    train_data = train_data.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocab})
    validation_data = validation_data.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocab})
    test_data = test_data.map(utils.map_tokens_to_int, fn_kwargs={"vocab": vocab})
    train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
    validation_data = validation_data.with_format(type="torch", columns=["ids", "label", "length"])
    test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])
  
    batch_size = 512
    # load data in batches of 512
    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(validation_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)
    

    # setting up parameters to initialize model
    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = len(train_data.unique("label"))
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5
    # init model
    model = LSTM(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    )

    # apply pretrained weights to model to improve accuracy and decrease training time
    model.apply(initialize_weights)
    vectors = torchtext.vocab.GloVe()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding
    lr = 5e-4

    # use adam optimization
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 10
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss, training_accuracy = train(
            train_data_loader, model, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(valid_data_loader, model, criterion, device)
        if validation_loss < best_valid_loss:
            best_valid_loss = validation_loss
            torch.save(model, "lstm.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {training_accuracy:.3f}")
        print(f"valid_loss: {validation_loss:.3f}, valid_acc: {validation_accuracy:.3f}")
