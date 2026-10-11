"""Project paths, immutable configuration and provenance helpers."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
_root = Path(os.environ.get('CFG_PROJECT_ROOT', '/mnt/data/LONG/hongxu/zjiaak')).expanduser()
if not _root.is_absolute():
    raise ValueError('CFG_PROJECT_ROOT must be an absolute directory')
BASE = _root.resolve()
BOUNDARIES = (BASE, (BASE / 'SSD').resolve(), (BASE / 'HDD').resolve())


def bounded(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) for root in BOUNDARIES):
        raise ValueError(f'Path outside project boundary: {path}')
    return path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def load_config(path=None):
    config = json.loads(Path(path or HERE / 'config.json').read_text())
    if config['model'] != 'JiT-B/16' or config['img_size'] != 256:
        raise ValueError('This experiment requires JiT-B/16 at 256x256')
    if config['steps'] not in (25, 50, 100) or config['num_images'] not in (10000, 50000):
        raise ValueError('Protocol requires 25/50/100 steps and 10k or 50k images')
    if config['class_num'] != 1000 or config['num_images'] % config['class_num']:
        raise ValueError('Invalid balanced ImageNet protocol')
    if config['batch_size'] <= 0:
        raise ValueError('Batch size must be positive')
    for key in ('checkpoint', 'fid_stats', 'inception_weights'):
        bounded(BASE / config[key])
    if config.get('noise'):
        bounded(BASE / config['noise'])
    groups = configured_groups(config)
    if not groups or len({g['name'] for g in groups}) != len(groups):
        raise ValueError('Group names must be unique and nonempty')
    for group in groups:
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', group['name']):
            raise ValueError('Invalid group folder name')
        if group['method'] not in ('euler', 'heun', 'heun_flip', 'ab2', 'cab2', 'sg', 'sg_prev') or group['cfg'] not in (1., 3.):
            raise ValueError('Invalid method/CFG')
        overrides = group.get('overrides', {})
        if set(overrides) - {'steps', 'cab_theta'}:
            raise ValueError('Only steps and cab_theta may vary between groups')
        effective = group_config(config, group)
        if effective['steps'] not in (25, 50, 100) or not 0 <= effective['cab_theta'] <= 1.5:
            raise ValueError('Invalid group steps/theta')
    if config.get('velocity_diagnostics') and config['cfg1_conditional_only']:
        raise ValueError('Velocity diagnostics require both conditional/unconditional branches')
    return config


def configured_groups(config):
    if 'groups' in config:
        return config['groups']
    return [dict(name=f'{method}_cfg{cfg:g}', method=method, cfg=cfg)
            for method in config['methods'] for cfg in config['cfg_values']]


def group_config(config, group):
    return dict(config, **group.get('overrides', {}))


def expected_nfe(method, steps):
    return 2 * steps - 1 if method in ('heun', 'heun_flip') else 2 * steps if method == 'sg' else steps


def environment(gpu):
    env = os.environ.copy()
    cache = BASE / 'HDD/cfg/cache'
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', TORCH_COMPILE_DISABLE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TORCH_HOME=str(cache), XDG_CACHE_HOME=str(cache),
               MPLCONFIGDIR=str(cache / 'matplotlib'))
    env.pop('FDS_GD_EAGER', None)
    env.pop('FDS_LOCAL_METRIC', None)
    return env


def code_hashes():
    return {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json')}


def git_revision():
    result = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def assert_upstream(config):
    for relative, expected in config['upstream_sha256'].items():
        if digest(HERE / 'upstream' / relative) != expected:
            raise RuntimeError(f'Upstream snapshot changed: {relative}')
