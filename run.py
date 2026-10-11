"""One invocation = one fresh run folder; launch groups on explicit GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import subprocess
import sys
from pathlib import Path

from common import BASE, HERE, assert_upstream, bounded, code_hashes, configured_groups, digest, environment, load_config, write_json
from baselines import existing_baselines


def validate(root):
    from PIL import Image, ImageDraw
    plan = json.loads((root / 'plan.json').read_text())
    rows = json.loads((root / 'samples.json').read_text())
    hashes = None
    results = []
    counts = {}
    for row in rows:
        counts[row['class_id']] = counts.get(row['class_id'], 0) + 1
    if plan['mode'] == 'formal' and (len(counts) != 1000 or set(counts.values()) != {plan['config']['num_images']//1000}):
        raise AssertionError('Class counts are not balanced')
    for group in plan['groups']:
        folder = root / group['name']
        noise = json.loads((folder / 'noise_hashes.json').read_text())
        if hashes is None:
            hashes = noise
        elif hashes != noise:
            raise AssertionError('Initial noise differs between groups')
        paths = sorted((folder / 'images').glob('*.png'))
        if [p.name for p in paths] != [f"{r['sample_id']:06d}.png" for r in rows]:
            raise AssertionError(f"Image set mismatch: {group['name']}")
        for path in paths:
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != 'RGB':
                    raise AssertionError('Image size/mode mismatch')
                image.verify()
        generation = json.loads((folder / 'generation.json').read_text())
        if plan['config'].get('velocity_diagnostics'):
            geometry = json.loads((folder / 'velocity_geometry/summary.json').read_text())
            if geometry['num_images'] != len(rows) or geometry['steps'] != generation['steps']:
                raise AssertionError('Velocity geometry coverage mismatch')
        metrics = json.loads((folder / 'metrics.json').read_text()) if (folder / 'metrics.json').exists() else None
        results.append(dict(group=group, generation=generation, metrics=metrics))
    if plan['mode'] == 'smoke':
        chosen = [rows[i]['sample_id'] for i in sorted({0, len(rows)//3, 2*len(rows)//3, len(rows)-1})]
        width, height = 180+128*len(chosen), 30+128*len(plan['groups'])
        sheet=Image.new('RGB',(width,height),'white')
        draw=ImageDraw.Draw(sheet)
        for col,sample_id in enumerate(chosen):
            draw.text((180+128*col,8),f'ID {sample_id:06d}',fill='black')
        for row,group in enumerate(plan['groups']):
            draw.text((8,30+128*row+50),group['name'],fill='black')
            for col,sample_id in enumerate(chosen):
                with Image.open(root/group['name']/'images'/f'{sample_id:06d}.png') as image:
                    sheet.paste(image.resize((128,128)),(180+128*col,30+128*row))
        sheet.save(root/'preview.png')
    return dict(mode=plan['mode'], image_count_per_group=len(rows), identical_initial_noise=True,
                class_counts=counts, results=results,external_baselines=existing_baselines(plan['config']))


def launch(args):
    c = load_config(args.config)
    assert_upstream(c)
    for key in ('checkpoint', 'fid_stats', 'inception_weights'):
        if digest(BASE / c[key]) != c[key + '_sha256']:
            raise RuntimeError(f'Asset hash mismatch: {key}')
    if c.get('noise') and digest(BASE/c['noise'])!=c['noise_sha256']:
        raise RuntimeError('Shared noise bank hash mismatch')
    baselines=existing_baselines(c,verify_images=args.mode=='smoke')
    if args.mode == 'formal':
        if not args.smoke:
            raise ValueError('Formal generation requires --smoke PATH to a completed smoke run')
        smoke = bounded(args.smoke)
        if (smoke / 'status.txt').read_text().strip() != 'SMOKE_COMPLETE':
            raise RuntimeError('Smoke run not complete')
        old = json.loads((smoke / 'plan.json').read_text())
        if old['config'] != c or old['code_sha256'] != code_hashes():
            raise RuntimeError('Config/code changed since smoke test; rerun smoke')
        if old['generation_batch_size'] != c['batch_size']:
            raise RuntimeError('Smoke must use the configured full batch size')
        smoke_names = {g['name'] for g in old['groups']}
        required = {g['name'] for g in configured_groups(c)}
        if not required.issubset(smoke_names):
            raise RuntimeError('Smoke run did not cover every configured group')
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError('Invalid GPU list')
    if args.smoke_images <= 0 or args.smoke_images > c['num_images'] or args.smoke_batch_size <= 0:
        raise ValueError('Invalid smoke count/batch')
    parent = BASE / ('HDD/cfg/smoketest' if args.mode == 'smoke' else 'HDD/cfg/runs')
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    root = parent / (args.mode + '_' + timestamp)
    root.mkdir(exist_ok=False)
    count = args.smoke_images if args.mode == 'smoke' else c['num_images']
    # Smoke samples cover the whole label space; the 10k manifest is balanced.
    ids = [i * (c['num_images'] - 1) // max(count - 1, 1) for i in range(count)] if args.mode == 'smoke' else list(range(count))
    groups = configured_groups(c)
    plan = dict(mode=args.mode, config=c, groups=groups, num_images=count,
                generation_batch_size=args.smoke_batch_size if args.mode == 'smoke' else c['batch_size'],
                code_sha256=code_hashes(), gpus=gpu_ids, smoke_reference=str(args.smoke),
                created_hong_kong=timestamp)
    write_json(root / 'plan.json', plan)
    write_json(root / 'reused_baselines.json',baselines)
    write_json(root / 'samples.json', [dict(sample_id=i, class_id=i//(c['num_images']//1000),
               seed=c['eval_seed'] if c.get('noise') else c['eval_seed']+i,
               noise_index=i*c.get('noise_index_stride',1)) for i in ids])
    (root / 'status.txt').write_text('RUNNING\n')
    print('RUN_ROOT', root, flush=True)
    lanes = [groups[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    def lane(gpu, selected):
        for group in selected:
            name = group['name']
            with (root / (name + '.log')).open('x') as log:
                result = subprocess.run([sys.executable, str(HERE / 'worker.py'), '--root', str(root), '--group', name],
                                        cwd=HERE, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} generation failed; see {root / (name + ".log")}')
            if args.mode == 'formal':
                with (root / (name + '_metrics.log')).open('x') as log:
                    result = subprocess.run([sys.executable, str(HERE / 'evaluate.py'), '--root', str(root), '--group', name],
                                            cwd=HERE, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    (root / name / 'status.txt').write_text('METRICS_FAILED\n')
                    raise RuntimeError(f'{name} evaluation failed')
            print('GROUP_COMPLETE', name, 'GPU', gpu, flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(lane, gpu, selected) for gpu, selected in zip(gpu_ids, lanes)]
            # Wait for every already launched lane even if another lane fails.
            failures = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
            if failures:
                raise RuntimeError('; '.join(failures))
        summary = validate(root)
        write_json(root / 'summary.json', summary)
        if args.mode == 'formal':
            lines = ['| Method | CFG | Steps | NFE | FID-10k | images/s | Sampling seconds |',
                     '|---|---:|---:|---:|---:|---:|---:|']
            for row in summary['external_baselines']:
                fid=row['metrics']['metrics']['frechet_inception_distance']
                lines.append(f"| Heun (existing) | {row['cfg']:g} | 50 | 99 | {fid:.6f} | historical | historical |")
            for row in summary['results']:
                g, p, m = row['group'], row['generation'], row['metrics']['metrics']
                lines.append(f"| {g['name']} | {g['cfg']:g} | {p['steps']} | {p['nfe_per_sample']} | "
                             f"{m['frechet_inception_distance']:.6f} | {p['images_per_sampling_second']:.3f} | {p['sampling_seconds']:.3f} |")
            (root / 'results.md').write_text('\n'.join(lines) + '\n')
        (root / 'status.txt').write_text('SMOKE_COMPLETE\n' if args.mode == 'smoke' else 'COMPLETE\n')
        print('RUN_COMPLETE', root, flush=True)
    except BaseException:
        (root / 'status.txt').write_text('FAILED\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'formal', 'validate'))
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--smoke-images', type=int, default=16)
    parser.add_argument('--smoke-batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.mode == 'validate':
        if args.root is None:
            parser.error('validate requires --root')
        print(json.dumps(validate(bounded(args.root)), indent=2))
    else:
        launch(args)
