"""Example of using the RNN code generator."""

import matplotlib.pyplot as plt
import pandas as pd
import time
import torch

from deep_cash.algorithm_env import create_algorithm_env
from deep_cash.rnn_code_generator import (
    CodeGeneratorRNN, create_training_data, random_training_example,
    generate_samples, train, time_since, n_metafeatures, n_characters,
    dropout_rate, n_rnn_layers, n_hidden, learning_rate)


# training parameters
n_training_samples = 50000
n_iters = 20000
print_every = 1000
plot_every = 1000
all_losses = []


algorithm_env = create_algorithm_env()
rnn = CodeGeneratorRNN(
    n_metafeatures, n_characters, n_hidden, n_characters,
    dropout_rate=dropout_rate, num_rnn_layers=n_rnn_layers)
optim = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
training_data = create_training_data(algorithm_env, n=n_training_samples)
total_loss = 0

start = time.time()

run_metadata = []
for i in range(1, n_iters + 1):
    train_index = i % len(training_data)  # loop over training data
    output, loss = train(
        rnn, optim, *random_training_example(*training_data[train_index]))
    total_loss += loss

    if i % print_every == 0:
        samples_string = ", ".join(generate_samples(rnn))
        print("%s (%d %d%%) %.4f" % (
            time_since(start), i, i / n_iters * 100, loss))
        print("samples: %s\n" % samples_string)
        run_metadata.append([i, loss, samples_string])
    if i % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

run_metadata = pd.DataFrame(
    run_metadata, columns=["iteration", "loss", "samples"])

print("final samples: %s" % ", ".join(generate_samples(rnn)))
torch.save(rnn.state_dict(), "artifacts/rnn_code_generator_model.pt")
run_metadata.to_csv("artifacts/rnn_code_generator_metadata.csv")
fig = plt.figure()
plt.plot(all_losses)
plt.show()
fig.savefig("artifacts/rnn_char_model_loss.png")
