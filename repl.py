import torch
import torchtext
import torchtext.data
import BagOfWords as bow
from BagOfWords import NBoW
import LSTM as lstm
from LSTM import LSTM
args = ('bow', 'lstm', 'exit', 'info')


tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
vocabulary = torch.load('vocabulary.pth')
# model = bow.NBoW(21478, 300, 2, bow.pad)
bow_model = torch.load('BagOfWords.pt')
bow_model.eval()

lstm_model = torch.load('lstm.pt')
lstm_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment = {0: "Negative", 1: "Positive"}

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def eval_cmd(str):

    if len(str) < 1 or len(str.split()) < 2:
        if str != 'quit' or str != 'info':
            print("Invalid command")
        if str == 'info':
            info()
        return
    
    cmd = str.split()[0]
    sentence = " ".join(str.split()[1:])

    if cmd not in args:
        print("Error: Command not found")
        return
    
    
    if cmd == 'bow':
        inference = bow.predict_sentiment(sentence, bow_model, tokenizer, vocabulary, device)
        
    if cmd == 'lstm':
        inference = lstm.predict_sentiment(sentence, lstm_model, tokenizer, vocabulary, device)
        print(inference)
    print(f"Predicted: {sentiment[inference[0]]} Probability: {inference[1]}")

def info():
    print("Welcome to the sentiment analysis repl. To enter a command, type the model architexture you")
    print("want to use followed by the sentence you want to evaluate. Enter \"quit\" to quit the repl")
    print("Example: bow I really liked this movie, I think it is great")
    print("Models: bow, lstm")
    print("--------------------------------------------------------------------------------------------")
    print()

    
arg = "info"
while arg != 'quit':
    arg = input()
    eval_cmd(arg)
    print()

