"""Recorded-data parity and combined phase plots for manuscript Figures 3 and 4."""
from pathlib import Path
import json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'reference_outputs'
OUT=ROOT.parent/'manuscript_figures/reference_images'
TIERS=['Simple (Binary)','Complex (Ternary)','Very Complex (Quinary)']
SHORT=['Binary','Ternary','Quinary'];COLORS=['#1b7837','#e08214','#b2182b']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':300})
num=pd.read_csv(DATA/'data_numeric_predictions.csv')
fig,axes=plt.subplots(2,2,figsize=(9.6,8.2))
for ax,(key,label) in zip(axes.flat,[('dS',r'$\Delta S_{mix}$ (J mol$^{-1}$ K$^{-1}$)'),('dH',r'$\Delta H_{mix}$ (kJ mol$^{-1}$)'),('VEC','VEC (e/atom)'),('delta',r'$\delta$ (%)')]):
    g=num[num.descriptor==key];y=g.reference.to_numpy();p=g.predicted.to_numpy()
    lo=min(y.min(),p.min());hi=max(y.max(),p.max());pad=.08*max(hi-lo,1e-9)
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#666666',lw=1)
    for t,c,s in zip(TIERS,COLORS,SHORT):
        v=g[g.tier==t];ax.scatter(v.reference,v.predicted,s=42,c=c,label=s,alpha=.8,edgecolor='white',linewidth=.4)
    rmse=np.sqrt(np.mean((p-y)**2));r2=1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
    ax.text(.05,.94,f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','alpha':.8,'edgecolor':'#cccccc'})
    ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Adopted reference: '+label,ylabel='Five-response mean: '+label)
    ax.grid(alpha=.2)
axes.flat[0].legend(loc='lower right',fontsize=8)
fig.tight_layout();fig.savefig(OUT/'Figure3_recorded_parity.png',bbox_inches='tight');plt.close(fig)

cm=pd.read_csv(DATA/'confusion_matrix.csv',index_col=0)
cp=pd.read_csv(DATA/'metrics_classification.csv').set_index('tier').loc[TIERS]
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1.1,1.2]})
ax=axes[0];im=ax.imshow(cm.to_numpy(),cmap='Blues',vmin=0,vmax=cm.to_numpy().max())
for i in range(len(cm)):
 for j in range(len(cm.columns)):
  v=int(cm.iloc[i,j]);ax.text(j,i,str(v),ha='center',va='center',color='white' if v>5 else 'black',fontweight='bold')
ax.set(xticks=range(len(cm.columns)),xticklabels=cm.columns,yticks=range(len(cm)),yticklabels=cm.index,xlabel='Predicted label',ylabel='Assigned reference label')
ax.set_title('(a) Phase-label confusion matrix',fontsize=11,fontweight='bold');fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Count')
ax=axes[1];x=np.arange(3);w=.36
for off,key,c,label in [(-w/2,'accuracy','#4575b4','Accuracy'),(w/2,'macro_F1','#d73027','Macro-F1')]:
 ax.bar(x+off,cp[key],w,color=c,label=label)
 for xx,v in zip(x+off,cp[key]):ax.text(xx,v+.02,f'{v:.3f}',ha='center',fontsize=9)
ax.set(xticks=x,xticklabels=SHORT,ylim=(0,1.12),ylabel='Score');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
ax.set_title('(b) Performance by complexity tier',fontsize=11,fontweight='bold')
fig.tight_layout();fig.savefig(OUT/'Figure4_recorded_classification.png',bbox_inches='tight');plt.close(fig)
print('Recorded-data manuscript Figures 3 and 4 generated.')
