# Copyright (c) Alibaba, Inc. and its affiliates.

from datasets import load_dataset

from evalscope.api.dataset.loader import _livecodebench_features
from evalscope.benchmarks.live_code_bench.load_utils import filter_date


def test_livecodebench_features_keep_contest_date_string(tmp_path):
    data_path = tmp_path / 'test.jsonl'
    data_path.write_text(
        '{"question_title":"A. Short Sort",'
        '"question_content":"question",'
        '"platform":"codeforces",'
        '"question_id":"1873_A",'
        '"contest_id":"1873",'
        '"contest_date":"2023-08-21T00:00:00",'
        '"starter_code":"",'
        '"difficulty":"easy",'
        '"public_test_cases":"[]",'
        '"private_test_cases":"[]",'
        '"metadata":"{}"}\n'
    )

    dataset = load_dataset(
        'json',
        data_files={'test': str(data_path)},
        split='test',
        features=_livecodebench_features(),
    )

    contest_date = dataset[0]['contest_date']
    assert isinstance(contest_date, str)
    assert filter_date(
        contest_date,
        start_date='2023-05-01',
        end_date='2025-04-30',
    )
