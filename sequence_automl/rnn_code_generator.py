"""An RNN to generated sklearn code.

TODO:
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


metafeature_categories = (
    ["not_executable", "executable"],
    ["not_creates_estimator", "creates_estimator"],
)
n_metafeatures = sum([len(m) for m in metafeature_categories])
characters = string.ascii_letters + string.digits + " .,;'-_()[]{}="
n_characters = len(characters) + 1  # add one for <EOS> token
n_hidden = 128  # number of hidden layers
criterion = nn.NLLLoss()  # negative log likelihood loss
learning_rate = 0.0005

# training parameters
n_iters = 5000
print_every = 500
plot_every = 500
all_losses = []

# dummy data for learning the sklearn API
# ["executable", "creates_estimator", "algorithm_string"]
automl_dummy_data = [
    (["executable", "not_creates_estimator"], "sklearn"),
    (["executable", "not_creates_estimator"], "sklearn.ensemble"),
    (["executable", "not_creates_estimator"], "sklearn.gaussian_process"),
    (["executable", "not_creates_estimator"], "sklearn.linear_model"),
    (["executable", "not_creates_estimator"], "sklearn.naive_bayes"),
    (["executable", "not_creates_estimator"], "sklearn.neighbors"),
    (["executable", "not_creates_estimator"], "sklearn.neural_network"),
    (["executable", "not_creates_estimator"], "sklearn.svm"),
    (["executable", "not_creates_estimator"], "sklearn.tree"),
]

class CodeGeneratorRNN(nn.Module):

    """Simple RNN module

    Adapted from:
    http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    """

    def __init__(
            self, metafeature_size, input_size, hidden_size, output_size,
            dropout_rate=0.1, num_rnn_layers=1):
        super(CodeGeneratorRNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.metafeature_size = metafeature_size
        self.rnn = nn.GRU(
            metafeature_size + input_size, hidden_size,
            num_layers=self.num_rnn_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, metafeatures, input, hidden):
        input_concat = torch.cat((metafeatures, input), 2)
        output, hidden = self.rnn(input_concat, hidden)
        output = self.dropout(self.decoder(output))
        output = output.view(output.shape[0], -1)  # dim <seq_length x n_chars>
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.num_rnn_layers, 1, self.hidden_size))


def character_to_index(c):
    """Get index of a character based on integer encoding."""
    return characters.find(c)


def create_metafeature_tensor(metafeatures, seq):
    """Convert a metafeature vector into a tensor.

    For now this will be a single category indicating `is_executable`.
    """
    m = []
    for i, f in enumerate(metafeatures):
        categories = metafeature_categories[i]
        t = torch.zeros(len(seq), 1, len(categories))
        cat_index = categories.index(f)
        for j, _ in enumerate(seq):
            t[j][0][cat_index] = 1
        m.append(t)
    m = torch.cat(m, 2)
    return m


def create_input_tensor(seq):
    """Convert a string of characters to an input tensor.

    Returns tensor of dim <string_length x 1 x n_characters>."""
    t = torch.zeros(len(seq), 1, n_characters)
    for i, c in enumerate(seq):
        t[i][0][character_to_index(c)] = 1
    return t


def create_target_tensor(seq):
    """Convert a string of character to a target tensor."""
    char_indices = [characters.find(seq[i]) for i in range(1, len(seq))]
    char_indices.append(n_characters - 1)  # add <EOS> token
    return torch.LongTensor(char_indices)


def random_datum(data):
    return data[random.randint(0, len(data) - 1)]


def random_training_example():
    """Sample a random input, target pair."""
    metafeatures, datum = random_datum(automl_dummy_data)
    metafeature_tensor = Variable(
        create_metafeature_tensor(metafeatures, datum))
    input_tensor = Variable(create_input_tensor(datum))
    target_tensor = Variable(create_target_tensor(datum))
    return metafeature_tensor, input_tensor, target_tensor


def train(rnn, optim, metafeature_tensor, input_tensor, target_tensor):
    """Train the character-level model."""
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0

    output, hidden = rnn(metafeature_tensor, input_tensor, hidden)
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


def sample(rnn, start_char, metafeatures, max_length=20):
    """Sample from generator given a starting character."""
    metafeature_tensor = Variable(
        create_metafeature_tensor(metafeatures, start_char))
    input_tensor = Variable(create_input_tensor(start_char))
    hidden = rnn.initHidden()

    output_sample = start_char

    for i in range(max_length):
        output, hidden = rnn(metafeature_tensor, input_tensor, hidden)
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
    metafeatures = [["executable", "not_creates_estimator"]
                    for _ in range(len(start_chars))]
    return [sample(rnn, c, m) for c, m in zip(start_chars, metafeatures)]


def main():
    rnn = CodeGeneratorRNN(
        n_metafeatures, n_characters, n_hidden, n_characters, num_rnn_layers=1)
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
            print("samples: %s\n" % samples_string)
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
