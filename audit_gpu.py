"""Real EMA1/BF16 Euler-Heun regression on exactly the same noise and labels."""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from common import BASE, HERE, assert_upstream, digest, load_config, write_json, bounded


def main(output_path):
    import torch
    from samplers import Sampler, initial_noise
    c = load_config()
    assert_upstream(c)
    if digest(BASE/c['checkpoint']) != c['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    sys.path.insert(0,str(HERE/'upstream'))
    from denoiser import Denoiser
    args=SimpleNamespace(model=c['model'],img_size=256,class_num=1000,attn_dropout=0.,proj_dropout=0.,
        label_drop_prob=.1,P_mean=-.8,P_std=.8,t_eps=.05,noise_scale=1.,ema_decay1=.9999,
        ema_decay2=.9996,sampling_method='heun',num_sampling_steps=50,cfg=3.,interval_min=.1,interval_max=1.)
    torch.set_num_threads(4)
    model=Denoiser(args)
    checkpoint=torch.load(BASE/c['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint[c['ema']],strict=True)
    del checkpoint
    model=model.cuda().eval().requires_grad_(False)
    noise=initial_noise([0,9999],c).cuda()
    labels=torch.tensor([0,999],device='cuda')
    results=[]
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        for cfg in (1.,3.):
            model.cfg_scale=cfg
            sampler=Sampler(model.net,c,cfg)
            for method in ('euler','heun'):
                model.method=method
                expected=model.generate(labels,None,noise=noise)
                actual=sampler.generate(noise,labels,method)
                equal=torch.equal(actual,expected)
                results.append(dict(cfg=cfg,method=method,exact_equal=equal,
                                    max_abs_difference=float((actual-expected).abs().max())))
                print(results[-1],flush=True)
                if not equal:
                    raise AssertionError(results[-1])
        # CFG1 equivalence at a fixed batch shape, without changing forward batching.
        t=torch.full((2,1,1,1),.98,device='cuda')
        velocity,clean=Sampler(model.net,c,1).guided_velocity(noise,t,labels)
        conditional=model.net(torch.cat((noise,noise)),torch.cat((t.flatten(),t.flatten())),
                         torch.cat((labels,torch.full_like(labels,1000))))[:2].float()
        if not torch.equal(velocity,(conditional-noise)/(1-t).clamp_min(c['t_eps'])):
            raise AssertionError('CFG1 conditional arithmetic mismatch')
    write_json(bounded(output_path),dict(checkpoint_sha256=c['checkpoint_sha256'],
               real_ema_bf16_regressions=results,cfg1_conditional_formula_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    main(args.output)
