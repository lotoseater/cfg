"""Replay only the first full batch and compare pixels to old Heun PNGs."""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
os.environ['TORCH_COMPILE_DISABLE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from common import BASE,HERE,assert_upstream,bounded,digest,load_config,write_json
from baselines import existing_baselines


def main(output):
    import numpy as np
    from PIL import Image
    import torch
    from samplers import Sampler,initial_noise
    c=load_config()
    assert_upstream(c)
    baselines=existing_baselines(c)
    if digest(BASE/c['checkpoint'])!=c['checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    sys.path.insert(0,str(HERE/'upstream'))
    from denoiser import Denoiser
    args=SimpleNamespace(model=c['model'],img_size=256,class_num=1000,attn_dropout=0.,proj_dropout=0.,
        label_drop_prob=.1,P_mean=-.8,P_std=.8,t_eps=.05,noise_scale=1.,ema_decay1=.9999,
        ema_decay2=.9996,sampling_method='heun',num_sampling_steps=50,cfg=3.,interval_min=.1,interval_max=1.)
    torch.set_num_threads(4)
    model=Denoiser(args)
    ckpt=torch.load(BASE/c['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt[c['ema']],strict=True)
    del ckpt
    model=model.cuda().eval().requires_grad_(False)
    ids=list(range(c['batch_size']))
    noise=initial_noise(ids,c).cuda()
    labels=torch.tensor([i//10 for i in ids],device='cuda')
    results=[]
    for baseline in baselines:
        sampler=Sampler(model.net,c,baseline['cfg'])
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
            generated=sampler.generate(noise,labels,'heun')
        arrays=np.round(np.clip(((generated.float()+1)/2).cpu().numpy()*255,0,255)).astype(np.uint8)
        exact=0; different_pixels=0; max_delta=0
        for sample_id,array in zip(ids,arrays):
            with Image.open(Path(baseline['source_images'])/f'{sample_id:05d}.png') as image:
                old=np.array(image)
            new=array.transpose(1,2,0)
            exact+=int(np.array_equal(old,new))
            different_pixels+=int(np.count_nonzero(old!=new))
            max_delta=max(max_delta,int(np.abs(old.astype(np.int16)-new.astype(np.int16)).max()))
        row=dict(name=baseline['name'],checked_samples=len(ids),exact_png_pixels_images=exact,
                 different_channel_values=different_pixels,max_uint8_difference=max_delta,
                 batch_size=c['batch_size'],checkpoint_sha256=c['checkpoint_sha256'],noise_sha256=c['noise_sha256'])
        results.append(row)
        print(row,flush=True)
        if exact!=len(ids):
            write_json(bounded(output),dict(status='MISMATCH',results=results))
            raise AssertionError('Historical PNG replay mismatch')
    write_json(bounded(output),dict(status='EXACT',results=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    main(parser.parse_args().output)
