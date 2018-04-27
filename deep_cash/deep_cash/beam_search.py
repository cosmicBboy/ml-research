"""Beam search implementation.

TODO:
- incorporate diversity into the rank score so that samples are differet.
"""

import torch
from torch.autograd import Variable
from copy import copy

from .rnn_code_generator import (
    load_model, create_metafeature_tensor, create_input_tensor,
    sos_index, sos_token, eos_index, eos_token, characters)


START_SAMPLE = ""


class BeamSearchDecoder(object):
    """Object to handle beam search logic."""

    def __init__(self, rnn, top_n=3):
        """Initialize BeamSearch object.

        :param CodeGeneratorRNN rnn: an RNN object.
        :param int top_n: number of candidate sequences to consider.
        """
        self.rnn = rnn
        self.top_n = top_n
        self.top_n_expand = min(top_n, rnn.output_size)

    def sample(
            self, start_char=sos_token,
            metafeatures=["executable", "creates_estimator"],
            max_length=100):
        """Sample from generator given a starting character.

        :param str start_char: starting character, default: sos_token "^"
        :param list[str] metafeatures: categorical values specified by
            rnn_code_generate.metafeature_categories.
        """
        metafeature_tensor = Variable(
            create_metafeature_tensor(metafeatures, start_char))
        hidden = self.rnn.initHidden()
        start_candidate = Candidate(0, [sos_index], start_char, hidden)
        candidates = self._expand_candidate(
            start_candidate, metafeature_tensor)

        for _ in range(max_length):
            new_candidates = []
            for c in candidates:
                new_candidates.extend(
                    self._expand_candidate(c, metafeature_tensor))
            candidates = self._prune_candidates(new_candidates)
        return candidates

    def _expand_candidate(self, candidate, metafeature_tensor):
        """Evaluates a candidate sequence."""
        # early return if the last char in sequence is eos token.
        if candidate.sequence[-1] == eos_token:
            return [candidate]
        output, hidden = self.rnn(
            metafeature_tensor,
            Variable(create_input_tensor(candidate.sequence[-1])),
            candidate.hidden_state)
        # rnn uses LogSoftmax as output layer, so no need to compute log
        # probability here.
        top_p, top_i = output.data.topk(self.top_n_expand)
        top_p, top_i = top_p.view(-1).tolist(), top_i.view(-1).tolist()
        prob_scores, indices, sequences, hidden_copies = [], [], [], []
        for p, i in zip(top_p, top_i):
            prob_scores.append(candidate.prob_score + p)
            indices.append(candidate.indices + [i])
            sequences.append(candidate.sequence + characters[i])
            hidden_copies.append(hidden.clone())
        return [Candidate(*args) for args in
                zip(prob_scores, indices, sequences, hidden_copies)]

    def _prune_candidates(self, candidates):
        """Picks the top n from a list of candidates."""
        scores = torch.Tensor([c.prob_score for c in candidates])
        _, top_cand_index = scores.topk(self.top_n)
        return [c for i, c in enumerate(candidates) if i in top_cand_index]


class Candidate(object):

    def __init__(self, prob_score, indices, sequence, hidden_state):
        """Initialize a candidate sample used by the BeamSearch routine."""
        self.prob_score = prob_score
        self.indices = indices
        self.sequence = sequence
        self.hidden_state = hidden_state

    def normalized_prob_score(self):
        pass

    @property
    def sample_code(self):
        """Clean up sample code sequence."""
        code = copy(self.sequence)
        if code.startswith(sos_token):
            code = code[1:]
        if code.endswith(eos_token):
            code = code[:-1]
        return code
