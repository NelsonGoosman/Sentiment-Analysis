import torch
import torch.nn as nn

def tokenize_str(str, tokenizer, max_length):
    '''
    Parameters: str -> The string to tokenize
                tokenizer -> an object that tokenizes sentences
                max_length -> trims tokens to size max_length if greater than max_length
    Returns: A dictionary containing tokens of the string
    '''
    tokens = tokenizer(str["text"])[:max_length]
    return {"tokens": tokens}

def map_tokens_to_int(tokens, vocab):
    '''
    Paramaters: tokens -> a dictionary of tokens
                vocab -> the model vocabulary
    Returns: A dictionary where the input tokens have been converted from strings to integers 
    '''
    ids = vocab.lookup_indices(tokens["tokens"])
    return {"ids": ids}

# source for get_collate_fn and get_data_loader: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb
def get_collate_fn(pad_index):
    '''
    Creates a function to collate the data. This pads shorter tensors to a uniform length.
    It returns a function that pads a batch at pad_index
    '''
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
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