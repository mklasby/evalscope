"""AIME targets are strings whether the dataset stores the answer as text or as an integer."""
import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.mark.parametrize('answer', [70, '70'])
def test_aime_answer_becomes_string_target(answer) -> None:
    adapter = get_benchmark('aime25', TaskConfig(model='mock', datasets=['aime25']))

    sample = adapter.record_to_sample({'problem': 'Find the sum.', 'answer': answer})

    assert sample.target == '70'
