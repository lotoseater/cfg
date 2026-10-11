"""Read-only validation and provenance of the existing paired Heun-10k runs."""
import json
import math
from common import BASE, bounded, digest


def existing_baselines(config, verify_images=False):
    from PIL import Image
    result=[]
    for row in config['external_baselines']:
        folder=bounded(BASE/row['folder'])
        if (folder/'status.txt').read_text().strip()!='COMPLETE':
            raise RuntimeError(f'Historical Heun incomplete: {folder}')
        manifest=json.loads((folder/'manifest.json').read_text())
        plan=manifest['plan']
        metrics=json.loads((folder/'metrics.json').read_text())
        for key in ('model','img_size','steps','interval_min','interval_max','num_images','batch_size',
                    'noise_sha256','checkpoint_sha256','fid_stats_sha256','inception_weights_sha256'):
            if plan[key]!=config[key]:
                raise RuntimeError(f'Historical Heun mismatch: {row["name"]} {key}')
        if plan['noise']!=config['noise'] or manifest['group']['cfg']!=row['cfg']:
            raise RuntimeError('Heun input/guidance mismatch')
        if manifest['group']['method']!='heun' or metrics['num_images']!=10000:
            raise RuntimeError('Expected original Heun-10k')
        if plan.get('attention_backend')!='flash':
            raise RuntimeError('Historical attention backend differs')
        if metrics['reference_sha256']!=config['fid_stats_sha256'] or metrics['inception_weights_sha256']!=config['inception_weights_sha256']:
            raise RuntimeError('Historical evaluation assets differ')
        if metrics.get('rng_seed',0)!=config['metric_rng_seed']:
            raise RuntimeError('Historical IS RNG differs')
        if not all(math.isfinite(float(metrics['metrics'][key])) for key in
                   ('frechet_inception_distance','inception_score_mean','inception_score_std')):
            raise RuntimeError('Invalid historical metrics')
        images=bounded(metrics['images'])
        paths=sorted(images.glob('*.png'))
        if [p.name for p in paths]!=[f'{i:05d}.png' for i in range(10000)]:
            raise RuntimeError('Historical image IDs/count differ')
        sampling=json.loads((images.parent/'config.json').read_text())
        for key in ('noise_scale','t_eps'):
            if sampling[key]!=config[key]:
                raise RuntimeError(f'Historical sampler mismatch: {key}')
        if verify_images:
            for path in paths:
                with Image.open(path) as image:
                    image.verify()
        result.append(dict(name=row['name'],method='heun',cfg=row['cfg'],external=True,
                      reused_without_regeneration_or_reevaluation=True,source_folder=str(folder),
                      source_images=str(images),metrics=metrics,
                      metrics_sha256=digest(folder/'metrics.json'),manifest_sha256=digest(folder/'manifest.json'),
                      source_commit=manifest['source_commit'],source_diff=manifest['source_diff'],
                      nfe_per_sample=99,steps=50,num_images=10000))
    return result
