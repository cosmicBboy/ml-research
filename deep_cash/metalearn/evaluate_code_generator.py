"""Module for generating samples from a pre-trained RNN.

TODO:
- measure proportion of samples are: executable, evalutes to estimator object.
"""

from rnn_code_generator import load_model, generate_samples


def main(path="rnn_code_generator_model.pt"):
    """Generate samples from the rnn_code_generator.py module."""
    rnn = load_model(path)
    print("samples: executable, creates_estimator")
    for _ in range(5):
        print(generate_samples(
            rnn, start_chars="^",
            metafeatures=[["executable", "creates_estimator"]]))
    print("\nsamples: not_executable, not_creates_estimator")
    for _ in range(5):
        print(generate_samples(
            rnn, start_chars="^",
            metafeatures=[["not_executable", "not_creates_estimator"]]))


if __name__ == "__main__":
    main()
