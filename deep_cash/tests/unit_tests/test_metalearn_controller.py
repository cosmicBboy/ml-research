from metalearn.algorithm_space import AlgorithmSpace
from metalearn.metalearn_controller import MetaLearnController
from metalearn import components, utils

import io
import tempfile
import torch


METAFEATURE_SIZE = 3


def _a_space():
    return AlgorithmSpace(
        classifiers=None,
        regressors=None,
        hyperparam_with_none_token=False,
        random_state=100)


def _metalearn_controller(
        a_space,
        metafeature_size=METAFEATURE_SIZE,
        input_size=5,
        hidden_size=5,
        output_size=5,
        dropout_rate=0.2,
        num_rnn_layers=3):
    return MetaLearnController(
        metafeature_size=METAFEATURE_SIZE,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        a_space=a_space,
        dropout_rate=dropout_rate,
        num_rnn_layers=num_rnn_layers)


def test_controller_equality():
    """CASH controller equality and non-equality checks."""
    torch.manual_seed(1000)
    controller = _metalearn_controller(_a_space())
    assert utils.models_are_equal(controller, controller)

    torch.manual_seed(1000)
    assert utils.models_are_equal(controller, _metalearn_controller(_a_space()))

    # changing the random seed will lead to different weights
    torch.manual_seed(1001)
    assert not utils.models_are_equal(controller, _metalearn_controller(_a_space()))

    # changing the algorithm space will lead to inequality
    torch.manual_seed(1000)
    diff_a_space = AlgorithmSpace(
        classifiers=[components.classifiers.logistic_regression()])
    diff_controller = _metalearn_controller(diff_a_space)
    assert not utils.models_are_equal(controller, diff_controller)

    torch.manual_seed(1000)
    diff_a_space = AlgorithmSpace(
        regressors=[components.regressors.lasso_regression()])
    diff_controller = _metalearn_controller(diff_a_space)
    assert not utils.models_are_equal(controller, diff_controller)

    torch.manual_seed(1000)
    diff_a_space = AlgorithmSpace(random_state=101)
    diff_controller = _metalearn_controller(diff_a_space)
    assert not utils.models_are_equal(controller, diff_controller)

    # changing the cash config will lead to unequal controllers
    a_space = _a_space()
    params = {
        "input_size": 10,
        "hidden_size": 10,
        "output_size": 10,
        "dropout_rate": 0.5,
        "num_rnn_layers": 5,
    }
    for k, v in params.items():
        torch.manual_seed(1000)
        diff_controller = _metalearn_controller(a_space, **{k: v})
        assert not utils.models_are_equal(controller, diff_controller)


def test_save_load_tempfile():
    """Cash controller can be saved/loaded onto a file specified as path."""
    w_controller = _metalearn_controller(_a_space())
    with tempfile.TemporaryFile() as f:
        w_controller.save(f)
        f.seek(0)
        r_controller = MetaLearnController.load(f)
    assert utils.models_are_equal(r_controller, w_controller)


def test_save_load_buffer():
    """Cash controller can be saved/loaded onto a file-like object."""
    w_controller = _metalearn_controller(_a_space())
    fileobj = io.BytesIO()
    w_controller.save(fileobj)
    fileobj.seek(0)
    r_controller = MetaLearnController.load(fileobj)
    assert utils.models_are_equal(r_controller, w_controller)
