"""An RNN to generate sklearn code.

TODO: move this to its own project at top level `ml-research` root

IDEAS:
- instead of sampling the max probability prediction, sample based on the
  probability distribution of the softmax.
- use skip connections i.e. resnets.
- use an attention model to generate code.
"""

import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

from .algorithm_env import CHARACTERS


metafeature_categories = (
    ["not_executable", "executable"],
    ["not_creates_estimator", "creates_estimator"],
)
n_metafeatures = sum([len(m) for m in metafeature_categories])
sos_token = "^"  # start of sentence token, like regex
eos_token = "$"  # end of sentence token, like regex
characters = CHARACTERS + sos_token + eos_token
n_characters = len(characters)
eos_index = characters.find(eos_token)  # end of sequence index
sos_index = characters.find(sos_token)  # start of sequence index

# model hyperparameters
dropout_rate = 0.1
n_rnn_layers = 3  # number of hidden layers
n_hidden = 128  # number of units in hidden layer
criterion = nn.NLLLoss()  # negative log likelihood loss
learning_rate = 0.0005


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
        self.output_size = output_size
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


def create_training_data(algorithm_env, n=5):
    """Samples from algorithm env and creates 4 variants per sample.

    - executable code sample (Estimator class)
    - executable code sample (Estimator instance)
    - non-executable partially randomized code
    - non-executable fully randomized code
    """
    training_data = []
    for i in range(n):
        sample_data = algorithm_env.sample_algorithm_code()
        training_data.extend([
            sample_data,
            algorithm_env.algorithm_obj_to_instance(sample_data),
        ])
    return training_data


def create_metafeature_tensor(metafeatures, seq):
    """Convert a metafeature vector into a tensor.

    For now this will be a single category indicating `is_executable`.

    :returns Tensor: dim <string_length x 1 x total_num_categories>, where
        total_num_categories accounts for all categorical values of the
        metafeatures.
    """
    m = []
    for i, f in enumerate(metafeatures):
        t = torch.zeros(len(seq), 1, len(metafeature_categories[i]))
        cat_index = metafeature_categories[i].index(f)
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
        t[i][0][characters.find(c)] = 1
    return t


def create_target_tensor(seq):
    """Convert a string of character to a target tensor."""
    char_indices = [characters.find(seq[i]) for i in range(1, len(seq))]
    char_indices.append(eos_index)  # add <EOS> token
    return torch.LongTensor(char_indices)


def _prepend_sos_token(input_seq):
    """Add sos token to the beginning of sequence."""
    return sos_token + input_seq


def random_training_example(metafeatures, input_seq):
    """Sample a random input, target pair."""
    input_seq = _prepend_sos_token(input_seq)
    metafeature_tensor = Variable(
        create_metafeature_tensor(metafeatures, input_seq))
    input_tensor = Variable(create_input_tensor(input_seq))
    target_tensor = Variable(create_target_tensor(input_seq))
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


def sample_rnn(rnn, start_char, metafeatures, max_length=100):
    """Sample from generator given a starting character."""
    metafeature_tensor = Variable(
        create_metafeature_tensor(metafeatures, start_char))
    input_tensor = Variable(create_input_tensor(start_char))
    hidden = rnn.initHidden()

    output_sample = ""

    for i in range(max_length):
        output, hidden = rnn(metafeature_tensor, input_tensor, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == eos_index:  # <EOS> token
            break
        else:
            char = characters[topi]
            output_sample += char
        input_tensor = Variable(create_input_tensor(char))

    return output_sample


def generate_samples(rnn, start_chars=sos_token * 3, metafeatures=None):
    """"""
    if metafeatures is None:
        metafeatures = [
            ["executable", "not_creates_estimator"]
            for _ in range(len(start_chars))]
    return [sample_rnn(rnn, c, m) for c, m in zip(start_chars, metafeatures)]


def load_model(path):
    rnn = CodeGeneratorRNN(
        n_metafeatures, n_characters, n_hidden, n_characters,
        dropout_rate=dropout_rate, num_rnn_layers=n_rnn_layers)
    rnn.load_state_dict(torch.load(path))
    return rnn
