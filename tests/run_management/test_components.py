"""Test the components class."""

# pylint: disable=all
# Temporarily disable all pylint checkers during AST traversal to prevent crash.
# The imports checker crashes when resolving simplexity package imports due to a bug
# in pylint/astroid: https://github.com/pylint-dev/pylint/issues/10185
# pylint: enable=all
# Re-enable all pylint checkers for the checking phase. This allows other checks
# (code quality, style, undefined names, etc.) to run normally while bypassing
# the problematic imports checker that would crash during AST traversal.

import re
from unittest.mock import Mock

import pytest

from simplexity.activations.activation_tracker import ActivationTracker
from simplexity.generative_processes.generative_process import GenerativeProcess
from simplexity.logging.logger import Logger
from simplexity.metrics.metric_tracker import MetricTracker
from simplexity.persistence.model_persister import ModelPersister
from simplexity.run_management.components import Components


def test_get_none():
    components = Components()
    assert components.get_optimizer() is None


def test_get_none_with_key_raises_error():
    components = Components()
    with pytest.raises(KeyError, match="No predictive model found"):
        components.get_predictive_model("mock")


def test_get_unique_instance():
    logger = Mock(spec=Logger)
    components = Components(loggers={"mock": logger})
    assert components.get_logger() == logger


def test_get_multiple_instances_without_key_raises_error():
    persister_1 = Mock(spec=ModelPersister)
    persister_2 = Mock(spec=ModelPersister)
    components = Components(persisters={"mock_1": persister_1, "mock_2": persister_2})
    with pytest.raises(KeyError, match="No key provided and multiple persisters found"):
        components.get_persister()


def test_get_instance_with_key():
    persister_1 = Mock(spec=ModelPersister)
    persister_2 = Mock(spec=ModelPersister)
    components = Components(persisters={"mock_1": persister_1, "mock_2": persister_2})
    assert components.get_persister("mock_1") == persister_1
    assert components.get_persister("mock_2") == persister_2


def test_get_instance_with_ending_key():
    activation_tracker_1 = Mock(spec=ActivationTracker)
    activation_tracker_2 = Mock(spec=ActivationTracker)
    components = Components(
        activation_trackers={
            "activation_tracker.mock_1": activation_tracker_1,
            "activation_tracker.mock_2": activation_tracker_2,
        }
    )
    assert components.get_activation_tracker("mock_1") == activation_tracker_1
    assert components.get_activation_tracker("mock_2") == activation_tracker_2


def test_get_instance_with_ending_key_raises_error():
    activation_tracker_1 = Mock(spec=ActivationTracker)
    activation_tracker_2 = Mock(spec=ActivationTracker)
    components = Components(
        activation_trackers={
            "activation_tracker_1.mock": activation_tracker_1,
            "activation_tracker_2.mock": activation_tracker_2,
        }
    )
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Multiple activation trackers with key 'mock' found: "
            "['activation_tracker_1.mock', 'activation_tracker_2.mock']"
        ),
    ):
        components.get_activation_tracker("mock")


def test_get_instance_with_ending_instance_key():
    metric_tracker_1 = Mock(spec=MetricTracker)
    metric_tracker_2 = Mock(spec=MetricTracker)
    components = Components(
        metric_trackers={
            "metric_tracker.mock_1.instance": metric_tracker_1,
            "metric_tracker.mock_2.instance": metric_tracker_2,
        }
    )
    assert components.get_metric_tracker("mock_1") == metric_tracker_1
    assert components.get_metric_tracker("mock_2") == metric_tracker_2


def test_get_instance_with_ending_instance_key_raises_error():
    metric_tracker_1 = Mock(spec=MetricTracker)
    metric_tracker_2 = Mock(spec=MetricTracker)
    components = Components(
        metric_trackers={
            "metric_tracker_1.mock.instance": metric_tracker_1,
            "metric_tracker_2.mock.instance": metric_tracker_2,
        }
    )
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Multiple metric trackers with key 'mock.instance' found: "
            "['metric_tracker_1.mock.instance', 'metric_tracker_2.mock.instance']"
        ),
    ):
        components.get_metric_tracker("mock")


def test_get_instance_with_no_matching_key_raises_error():
    generative_process = Mock(spec=GenerativeProcess)
    components = Components(generative_processes={"generative_process.mock": generative_process})
    with pytest.raises(KeyError, match=re.escape("Generative process with key 'does_not_exist' not found")):
        components.get_generative_process("does_not_exist")
