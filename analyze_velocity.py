"""Plot immutable per-step geometry summaries; no server/model access."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=HERE/'velocity_curves_20261011.json')
parser.add_argument('--output',type=Path,default=HERE/'velocity_analysis_20261011')
args = parser.parse_args()
DATA = json.loads(args.input.read_text())
OUT = args.output
OUT.mkdir(exist_ok=True)
COLORS = dict(ab2='#2471a3', cab2='#8951a5', sg='#d45a21', heun_flip='#4e6468')
LABELS = dict(ab2='AB2 / 100 steps', cab2='CAB2 theta=0.9 / 100 steps',
              sg='SG shift=0.05 / 50 steps', heun_flip='Heun-flip / 50 steps')
STYLES = dict(ab2='-', cab2='--', sg='-', heun_flip=':')
plt.rcParams.update({'font.size':10, 'axes.titlesize':13, 'axes.labelsize':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.fonttype':'none', 'figure.facecolor':'white'})


def curve(group, key, stat='mean'):
    rows=group['rows']
    return np.asarray([r['time'] for r in rows]), np.asarray([
        r['stats'].get(key,{}).get(stat) if r['stats'].get(key,{}).get(stat) is not None else np.nan
        for r in rows],dtype=float)


def middle_mean(group,key):
    t,v=curve(group,key)
    selected=v[(t>=.2-1e-6)&(t<=.9+1e-6)]
    return float(np.nanmean(selected)) if np.isfinite(selected).any() else None


def decorate(ax, cfg, title, ylabel, ylim=None):
    ax.set_title(f'{title} | CFG={cfg}')
    ax.set_xlabel('t (noise 0 -> data 1)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,1)
    if ylim:ax.set_ylim(*ylim)
    ax.axvspan(.95,1,color='#f3c5c5',alpha=.4,zorder=0)
    ax.axvline(.1,color='#777777',linestyle='--',linewidth=.75,alpha=.55)
    ax.grid(alpha=.18)


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


fig,axes=plt.subplots(3,2,figsize=(12,10.4))
panels=[('current.cond_vs_uncond.angle_deg','Conditional vs unconditional','Angle (degrees)',(0,20),1),
        ('correction_vs_base.norm_ratio_a_over_b','Method correction relative to base','100 * ||d|| / ||v_CFG|| (%)',(0,100),100),
        ('combined_vs_base.angle_deg','Combined velocity vs base','Angle (degrees)',(0,60),1)]
for col,cfg in enumerate((1,3)):
    for row,(key,title,ylabel,ylim,scale) in enumerate(panels):
        ax=axes[row,col]
        for g in DATA['groups']:
            if g['cfg']!=cfg:continue
            t,v=curve(g,key)
            ax.plot(t,v*scale,STYLES[g['method']],color=COLORS[g['method']],lw=1.8,label=LABELS[g['method']])
        decorate(ax,cfg,title,ylabel,ylim)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.958),ncol=2,frameon=False)
fig.suptitle('Velocity geometry | JiT-B/16 | ImageNet-10k',fontsize=17,y=.995)
fig.text(.5,.015,'Means over 10,000 images on each method\'s own trajectory. d = combined velocity - raw CFG velocity.\n'
         'Red shade: t >= 0.95 denominator clamp. Dashed vertical line: CFG interval boundary. Heun-flip tail is Euler.',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.055,1,.9))
save(fig,'velocity_geometry_overview')

fig,axes=plt.subplots(2,2,figsize=(12,7.8))
for col,cfg in enumerate((1,3)):
    for g in DATA['groups']:
        if g['cfg']!=cfg:continue
        t,c=curve(g,'current.cond.rms');_,u=curve(g,'current.uncond.rms')
        axes[0,col].plot(t,c,STYLES[g['method']],color=COLORS[g['method']],lw=1.8,label=LABELS[g['method']])
        axes[0,col].plot(t,u,':',color=COLORS[g['method']],lw=.8,alpha=.75)
        t,ratio=curve(g,'current.cond_vs_uncond.norm_ratio_a_over_b')
        axes[1,col].plot(t,ratio,STYLES[g['method']],color=COLORS[g['method']],lw=1.8)
    decorate(axes[0,col],cfg,'Basic branch magnitudes','RMS velocity',(.25,1.25))
    decorate(axes[1,col],cfg,'Conditional / unconditional norm ratio','||v_cond|| / ||v_uncond||',(.95,1.025))
    axes[1,col].axhline(1,color='black',lw=.7,alpha=.5)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.956),ncol=2,frameon=False)
fig.suptitle('Basic conditional / unconditional velocity magnitudes',fontsize=16,y=.995)
fig.text(.5,.014,'Conditional lines: method style; unconditional: thin dotted lines in the same color (almost overlap).\n'
         'RMS = L2 / sqrt(3*256*256). Ratios are per-image ratios averaged over 10k images.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.06,1,.89))
save(fig,'basic_branch_magnitudes')

fig,axes=plt.subplots(2,2,figsize=(12,7.6))
for col,cfg in enumerate((1,3)):
    for g in DATA['groups']:
        if g['cfg']!=cfg:continue
        if g['method']!='heun_flip':
            key='shifted.current_vs_aux.cfg.angle_deg' if g['method']=='sg' else 'history.cfg.angle_deg'
            t,v=curve(g,key)
            detail='same x, dt=0.05' if g['method']=='sg' else 'past x and t, dt=0.01'
            axes[0,col].plot(t,v,STYLES[g['method']],color=COLORS[g['method']],lw=1.7,
                             label=LABELS[g['method']].split(' / ')[0]+' ('+detail+')')
    decorate(axes[0,col],cfg,'Raw velocity pair angle','Angle (degrees)',(0,45))
    groups=[g for g in DATA['groups'] if g['cfg']==cfg]
    pos=np.arange(len(groups));w=.34
    par=[100*middle_mean(g,'correction.parallel_over_base') for g in groups]
    perp=[100*middle_mean(g,'correction.perpendicular_over_base') for g in groups]
    ax=axes[1,col]
    ax.bar(pos-w/2,par,w,color='#364c61',label='Parallel (signed)')
    ax.bar(pos+w/2,perp,w,color='#d4984d',label='Perpendicular magnitude')
    ax.set_xticks(pos,['AB2','CAB2','SG','Heun-flip'])
    ax.set_title(f'Correction decomposition, 0.2 <= t <= 0.9 | CFG={cfg}')
    ax.set_ylabel('Relative to ||v_CFG|| (%)');ax.grid(axis='y',alpha=.18)
    ax.axhline(0,color='black',lw=.7)
    for x,a,b in zip(pos,par,perp):
        ax.text(x-w/2,a+.1,f'{a:.2f}',ha='center',fontsize=8)
        ax.text(x+w/2,b+.1,f'{b:.2f}',ha='center',fontsize=8)
    ax.legend(frameon=False,fontsize=9)
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Reference velocity pairs and correction directions',fontsize=16,y=.99)
fig.text(.5,.015,'Different time offsets and state arguments: pair angles are descriptive, not an isolated shift/coefficient ablation.\n'
         'Bottom panels: first average per image, then average uniformly over each method\'s actual grid in [0.2, 0.9].',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.06,1,.95))
save(fig,'velocity_pairs_and_decomposition')

metrics=['current.cond_vs_uncond.angle_deg','current.cond_vs_uncond.norm_ratio_a_over_b',
         'combined_vs_base.angle_deg','combined_vs_base.norm_ratio_a_over_b',
         'correction_vs_base.norm_ratio_a_over_b','correction.parallel_over_base',
         'correction.perpendicular_over_base','correction_vs_cfg_direction.angle_deg',
         'cfg_direction_vs_base.angle_deg','cfg_correction_vs_base.norm_ratio_a_over_b',
         'cab.theta_vs_base.norm_ratio_a_over_b']
aggregates=[dict(name=g['name'],cfg=g['cfg'],method=g['method'],
                 interior_time_range=[.2,.9],means={k:middle_mean(g,k) for k in metrics}) for g in DATA['groups']]
(OUT/'interior_summary.json').write_text(json.dumps(aggregates,indent=2,allow_nan=False)+'\n')

lines=['# 速度几何分析：2026-10-11，100 NFE 轮次','',
       '源目录：`'+DATA['run_root']+'`。每组10000张；逐样本统计后按实际时间汇总。',
       'AB2/CAB2为100步，SG为50步且shift=.05/scale1，Heun-flip为50步/99 NFE；CFG1/3。',
       '大小使用CHW整体L2或RMS，两者的比值与角度一致。下表为0.2≤t≤0.9的时间均值，避免启动、CFG切换和末端截断主导结论。',
       '各方法沿自己的轨迹统计；网格不同，因此描述实际运行行为，不是共同状态上的严格机制消融。','',
       '![完整几何曲线](velocity_geometry_overview.png)','',
       '![条件/无条件大小](basic_branch_magnitudes.png)','',
       '![参照速度及修正分解](velocity_pairs_and_decomposition.png)','',
       '| 方法 | CFG | c/u夹角° | c/u大小比 | 修正/原速度 | 组合/原速度 | 组合转角° | 修正与CFG方向夹角° |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for g,a in zip(DATA['groups'],aggregates):
    m=a['means']
    lines.append(f"| {LABELS[g['method']]} | {g['cfg']:g} | {m['current.cond_vs_uncond.angle_deg']:.3f} | "
                 f"{m['current.cond_vs_uncond.norm_ratio_a_over_b']:.6f} | {100*m['correction_vs_base.norm_ratio_a_over_b']:.2f}% | "
                 f"{m['combined_vs_base.norm_ratio_a_over_b']:.4f} | {m['combined_vs_base.angle_deg']:.3f} | "
                 f"{m['correction_vs_cfg_direction.angle_deg']:.2f} |")
lines += ['', '基础条件/无条件速度主段大小几乎相等，夹角约1°。以AB2/CFG1的t=.5为例，RMS为1.144795和1.144447，夹角.819818°，大小比的P10–P90为.9996–1.0011。',
          'AB2/CAB2主段修正主要垂直于当前速度；SG同时有正的平行分量和较大的垂直分量，相对自己的原始CFG速度有约10%的放大。',
          '基础CFG方向为vc-vu；主段与原速度的夹角约87–90°。方法修正与该方向的夹角约75–89°，不足以说它们沿同一个CFG语义方向；它们共享的是差分外推的代数形式。',
          '', '| t=.98 | CFG1修正/原速度 | CFG3修正/原速度 | CFG1转角° | CFG3转角° |',
          '|---|---:|---:|---:|---:|']
for method in ('ab2','cab2','sg','heun_flip'):
    selected=[g for g in DATA['groups'] if g['method']==method]
    rows=[min(g['rows'],key=lambda r:abs(r['time']-.98)) for g in selected]
    corr=[100*r['stats']['correction_vs_base.norm_ratio_a_over_b']['mean'] for r in rows]
    angle=[r['stats']['combined_vs_base.angle_deg']['mean'] for r in rows]
    lines.append(f'| {LABELS[method]} | {corr[0]:.2f}% | {corr[1]:.2f}% | {angle[0]:.2f} | {angle[1]:.2f} |')
lines += ['', '末端影响明显增大。Heun-flip最后一段是Euler，故t=.98的Heun修正为0，不表示flip速度与normal相同。CAB2/CFG3在t=.99总修正为69.99%，转角33.36°，其中theta项为原速度的55.26%；t=.96总修正51.42%、转角30.56°。',
          'SG/CFG1在t=.98两条原始速度的夹角34.41°，组合转角42.96°；CFG3分别38.97°、35.18°。大修正与本轮较差FID并存，但还不能从这一次运行证明因果。',
          't>=.95触发JiT速度分母截断；SG两时间分支在末段可能使用不同分母，因此不能将末端行为完全归因于算法本身。',
          'SG/CFG3在t=.12/.14当前分支CFG=3、shifted分支CFG=1；t=.16两支均=3。这是既定CFG interval的结果，相关突变不是纯时间偏移效应。',
          '在未截断区间CAB的e=clean-v等于x-tv。截断后两者相差(max(1-t,.05)-(1-t))*v，所以末段e不能再严格等同于该坐标变换的导数。',
          '基础vc/vu的角度小不代表CFG无效：两条速度均含相同的-x项；差向量及其随时间积累仍可显著改变结果。',
          '', '建议后续优先：保持其它设置，缩小SG shift或scale；对CAB做仅末段theta消融；用共同状态诊断验证方法修正是否沿同一方向。当前分析没有自动启动任何新实验。',
          '', '完整每图统计仍在各组velocity_geometry/batch_*.npz，全部字段与均值/分位数在schema.json和summary.json。本报告只使用保存的summary，不重跑模型或重算FID。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print('OUTPUT',OUT)
print(json.dumps(aggregates,ensure_ascii=False))
