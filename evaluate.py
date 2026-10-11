"""FID-10k / IS with the same reference and precision as fds acceptance8."""
import argparse
import json
from pathlib import Path
from common import BASE, bounded, digest, write_json


def evaluate(root, name):
    import numpy as np
    import torch_fidelity
    from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id
    from torch_fidelity.metric_fid import fid_featuresdict_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_featuresdict_to_metric
    root = bounded(root)
    plan = json.loads((root / 'plan.json').read_text())
    c = plan['config']
    if plan['mode'] != 'formal':
        raise ValueError('Smoke images are not FID-10k results')
    folder = root / name
    if (folder / 'status.txt').read_text().strip() != 'GENERATED':
        raise RuntimeError('Group has not completed generation')
    if (folder / 'metrics.json').exists():
        raise FileExistsError(folder / 'metrics.json')
    if [p.name for p in sorted((folder / 'images').glob('*.png'))] != [f'{i:06d}.png' for i in range(c['num_images'])]:
        raise ValueError('Invalid complete image set')
    for key in ('fid_stats', 'inception_weights'):
        if digest(BASE / c[key]) != c[key + '_sha256']:
            raise RuntimeError(f'Asset hash mismatch: {key}')
    with np.load(BASE / c['fid_stats'], allow_pickle=False) as reference:
        # Match the reused baseline8 evaluator's float64 statistics exactly.
        stats = {k: np.array(reference[k],dtype=np.float64) for k in ('mu', 'sigma')}
    if stats['mu'].shape != (2048,) or stats['sigma'].shape != (2048, 2048):
        raise ValueError('Expected 2048-dimensional Inception statistics')
    kwargs = dict(input1=str(folder / 'images'), batch_size=c['metric_batch_size'], cuda=True,
                  verbose=True, cache=False, rng_seed=c['metric_rng_seed'], isc_splits=c['metric_isc_splits'],
                  feature_extractor_weights_path=str(BASE / c['inception_weights']))
    extractor = create_feature_extractor('inception-v3-compat', ['2048', 'logits_unbiased'], **kwargs)
    features = extract_featuresdict_from_input_id(1, extractor, **kwargs)
    generated = fid_featuresdict_to_statistics(features, '2048')
    generated = {key:value.astype(np.float64) for key,value in generated.items()}
    metrics = fid_statistics_to_metric(generated, stats, verbose=True)
    metrics.update(isc_featuresdict_to_metric(features, 'logits_unbiased', **kwargs))
    if not all(np.isfinite(float(v)) for v in metrics.values()):
        raise FloatingPointError('Non-finite evaluation metrics')
    write_json(folder / 'metrics.json', dict(label=f"FID-{c['num_images']//1000}k", metrics=metrics,
               num_images=c['num_images'], rng_seed=c['metric_rng_seed'], isc_splits=c['metric_isc_splits'],
               reference_sha256=c['fid_stats_sha256'], weights_sha256=c['inception_weights_sha256'],
               torch_fidelity=torch_fidelity.__version__))
    (folder / 'status.txt').write_text('COMPLETE\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    evaluate(args.root, args.group)
