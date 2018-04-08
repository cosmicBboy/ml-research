"""An RNN to generated sklearn code.

TODO:
- add a category input that encodes two variables:
  - is_executable {0, 1}
  - creates_Estimator {0, 1}
  - use this category input to conditionally sample from learned generator.
- use a “start of sentence” <SOS> token so that sampling can be done without
  choosing a start letter
- instead of sampling the max probability prediction, sample based on the
  probability distribution of the softmax.
"""

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import random
import sklearn
import string
import time
import torch
import torch.nn as nn
from torch.autograd import Variable


characters = string.ascii_letters + string.digits + " .,;'-_()[]{}="
n_characters = len(characters) + 1  # add one for <EOS> token
n_hidden = 128  # number of hidden layers
criterion = nn.NLLLoss()  # negative log likelihood loss
learning_rate = 0.0005

# training parameters
n_iters = 10000
print_every = 1000
plot_every = 1000
all_losses = []

# these data assume that all imports have been made
automl_data = [
    "sklearn",
    "sklearn.ensemble",
    "sklearn.gaussian_process",
    "sklearn.linear_model",
    "sklearn.naive_bayes",
    "sklearn.neighbors",
    "sklearn.neural_network",
    "sklearn.svm",
    "sklearn.tree",
]

class CodeGeneratorRNN(nn.Module):

    """Simple RNN module

    Adapted from:
    http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    """

    def __init__(
            self, input_size, hidden_size, output_size, dropout_rate=0.1,
            num_rnn_layers=1):
        super(CodeGeneratorRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size, hidden_size, num_layers=self.num_rnn_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(self.decoder(output))
        output = output.view(output.shape[0], -1)  # dim <seq_length x n_chars>
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))


def character_to_index(c):
    """Get index of a character based on integer encoding."""
    return characters.find(c)


def create_input_tensor(s):
    """Convert a string of characters to an input tensor.

    Returns tensor of dim <string_length x 1 x n_characters>."""
    t = torch.zeros(len(s), 1, n_characters)
    for i, c in enumerate(s):
        t[i][0][character_to_index(c)] = 1
    return t


def create_target_tensor(s):
    """Convert a string of character to a target tensor."""
    char_indices = [characters.find(s[i]) for i in range(1, len(s))]
    char_indices.append(n_characters - 1)  # add <EOS> token
    return torch.LongTensor(char_indices)


def random_datum(data):
    return data[random.randint(0, len(data) - 1)]


def random_training_example():
    """Sample a random input, target pair."""
    datum = random_datum(automl_data)
    input_tensor = Variable(create_input_tensor(datum))
    target_tensor = Variable(create_target_tensor(datum))
    return input_tensor, target_tensor


def train(rnn, optim, input_tensor, target_tensor):
    """Train the character-level model."""
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    output, hidden = rnn(input_tensor, hidden)
    output = output.view(output.shape[0], -1)
    loss += criterion(output, target_tensor)

    loss.backward()
    optim.step()

    return output, loss.data[0] / input_tensor.size()[0]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def sample(rnn, start_char="s", max_length=20):
    """Sample from generator given a starting character."""
    input_tensor = Variable(create_input_tensor(start_char))
    hidden = rnn.initHidden()

    output_sample = start_char

    for i in range(max_length):
        output, hidden = rnn(input_tensor, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == n_characters - 1:  # <EOS> token
            break
        else:
            char = characters[topi]
            output_sample += char
        input_tensor = Variable(create_input_tensor(char))

    return output_sample


def samples(rnn, start_chars="ssssss"):
    return [sample(rnn, start_char=c) for c in start_chars]


def main():
    rnn = CodeGeneratorRNN(
        n_characters, n_hidden, n_characters, num_rnn_layers=1)
    optim = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    total_loss = 0

    start = time.time()

    run_metadata = []

    for i in range(1, n_iters + 1):
        output, loss = train(rnn, optim, *random_training_example())
        total_loss += loss

        if i % print_every == 0:
            samples_string = ", ".join(samples(rnn))
            print("%s (%d %d%%) %.4f" % (
                time_since(start), i, i / n_iters * 100, loss))
            print("samples: %s" % samples_string)
            run_metadata.append([i, loss, samples_string])
        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    run_metadata = pd.DataFrame(
        run_metadata, columns=["iteration", "loss", "samples"])

    print("final samples: %s" % ", ".join(samples(rnn)))
    torch.save(rnn.state_dict(), "rnn_code_generator_model.pt")
    run_metadata.to_csv("rnn_code_generator_metadata.csv")
    fig = plt.figure()
    plt.plot(all_losses)
    plt.show()
    fig.savefig("rnn_char_model_loss.png")


if __name__ == "__main__":
    main()
