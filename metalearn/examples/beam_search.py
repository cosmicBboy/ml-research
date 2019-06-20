"""Example use of beam search object."""

from pathlib import Path
from os.path import dirname

from metalearn.beam_search import BeamSearchDecoder
from metalearn.rnn_code_generator import load_model


artifact_path = Path(dirname(__file__)) / "artifacts"
rnn = load_model(str(artifact_path / "rnn_code_generator_model.pt"))
beam = BeamSearchDecoder(rnn, top_n=20)
top_candidates = beam.sample(
    "^", ["executable", "creates_estimator"], max_length=100)
print("Beamsearch Samples:")
print("-------------------")
for c in top_candidates:
    print(c.sample_code)
