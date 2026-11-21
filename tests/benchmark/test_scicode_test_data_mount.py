"""SciCode mounts test_data.h5 read-only into its sandboxes instead of baking it into the image; a test_data_path
extra parameter names an existing copy, which is then not downloaded."""
import pytest

from evalscope.api.dataset import DatasetHub
from evalscope.api.registry import get_benchmark
from evalscope.config import SandboxTaskConfig, TaskConfig

MOUNT = {'bind': '/workspace/test_data.h5', 'mode': 'ro'}


def _adapter(test_data_path: str = ''):
    dataset_args = {'scicode': {'extra_params': {'test_data_path': test_data_path}}} if test_data_path else {}
    return get_benchmark(
        'scicode',
        TaskConfig(
            model='mock',
            datasets=['scicode'],
            dataset_args=dataset_args,
            sandbox=SandboxTaskConfig(enabled=True, engine='docker', default_config={'network_enabled': False}),
        ),
    )


def _parent_load(adapter):
    return next(cls for cls in type(adapter).__mro__[1:] if 'load' in vars(cls))


def test_given_test_data_is_mounted_read_only(tmp_path) -> None:
    data = tmp_path / 'test_data.h5'
    data.write_bytes(b'h5')

    adapter = _adapter(str(data))

    assert adapter._benchmark_meta.sandbox_config['volumes'] == {str(data.resolve()): MOUNT}
    # Another adapter does not inherit the mount: it mounts its own default copy.
    default_adapter = _adapter()
    assert default_adapter._benchmark_meta.sandbox_config['volumes'] == {
        str((default_adapter.docker_path / 'test_data.h5').resolve()): MOUNT
    }


def test_mount_reaches_the_docker_sandbox_config(tmp_path) -> None:
    pytest.importorskip('ms_enclave')
    from evalscope.api.sandbox import SandboxEngine, build_sandbox_config

    data = tmp_path / 'test_data.h5'
    data.write_bytes(b'h5')
    adapter = _adapter(str(data))

    merged = adapter._get_backend()._resolve_sandbox_config_dict()
    config = build_sandbox_config(SandboxEngine.DOCKER, merged)

    assert config.volumes[str(data.resolve())] == MOUNT
    assert merged['network_enabled'] is False


def test_given_test_data_is_not_downloaded(tmp_path, monkeypatch) -> None:
    data = tmp_path / 'test_data.h5'
    data.write_bytes(b'h5')
    adapter = _adapter(str(data))
    monkeypatch.setattr(DatasetHub, 'download_file', lambda self, file_path: pytest.fail('downloaded test data'))
    monkeypatch.setattr(_parent_load(adapter), 'load', lambda self: 'dataset')

    assert adapter.load() == 'dataset'


def test_missing_test_data_path_is_an_error(tmp_path, monkeypatch) -> None:
    adapter = _adapter(str(tmp_path / 'missing.h5'))
    monkeypatch.setattr(_parent_load(adapter), 'load', lambda self: 'dataset')

    with pytest.raises(FileNotFoundError, match='missing.h5'):
        adapter.load()


def test_without_test_data_path_the_hub_copy_is_placed_at_the_mounted_path(tmp_path, monkeypatch) -> None:
    downloaded = tmp_path / 'hub' / 'test_data.h5'
    downloaded.parent.mkdir()
    downloaded.write_bytes(b'h5 from the hub')
    adapter = _adapter()
    adapter.test_data_path = tmp_path / 'docker' / 'test_data.h5'
    adapter.test_data_path.parent.mkdir()
    requested = []

    def download_file(self, file_path):
        requested.append((self.data_id_or_path, file_path))
        return str(downloaded)

    monkeypatch.setattr(DatasetHub, 'download_file', download_file)
    monkeypatch.setattr(_parent_load(adapter), 'load', lambda self: 'dataset')

    assert adapter.load() == 'dataset'
    assert requested == [(adapter.dataset_id, 'test_data.h5')]
    assert adapter.test_data_path.read_bytes() == b'h5 from the hub'
