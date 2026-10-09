"""Regenerates docs/distinctive_words.png from data/company-document-text.csv. Run from the repository root: python docs/make_figure.py"""
import matplotlib as mpl
S=['#2a78d6','#eb6834','#1baf7a','#eda100']
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; SURF='#ffffff'
mpl.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,
 'font.family':'DejaVu Sans','font.size':11,'axes.edgecolor':GRID,'axes.labelcolor':INK2,
 'xtick.color':INK2,'ytick.color':INK2,'axes.titlecolor':INK,'axes.titlesize':13,'axes.titleweight':'bold',
 'axes.titlelocation':'left','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,
 'grid.linewidth':0.8,'axes.axisbelow':True,'legend.frameon':False,'legend.labelcolor':INK2})
import re, collections
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
d=pd.read_csv('data/company-document-text.csv')
classes=['invoice','purchase Order','ShippingOrder','report']
tok=lambda s: set(re.findall(r'[a-z]+',str(s).lower()))
docsets={c:[tok(t) for t in d[d.label==c].text] for c in classes}
df={c:collections.Counter(w for s in docsets[c] for w in s) for c in classes}
n={c:len(docsets[c]) for c in classes}
share=lambda c,w: df[c][w]/n[c]
words=[]; sizes=[]
for c in classes:
    sc={w:share(c,w)-max(share(o,w) for o in classes if o!=c) for w in df[c] if len(w)>2}
    words+= [w for w in sorted(sc,key=lambda w:(-round(sc[w],2),w)) if sc[w]>0.5][:4]
    sizes.append(len(words))
M=np.array([[100*share(c,w) for c in classes] for w in words])
cmap=LinearSegmentedColormap.from_list('b',['#f4f8fd','#2a78d6','#123f75'])
fig,ax=plt.subplots(figsize=(7.6,0.36*len(words)+1.7))
ax.imshow(M,cmap=cmap,vmin=0,vmax=100,aspect='auto')
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xticks(range(4),[f'{c}\n(n = {n[c]})' for c in classes]); ax.xaxis.tick_top()
ax.set_yticks(range(len(words)),words); ax.tick_params(length=0,labelcolor=INK)
for i in range(len(words)):
    for j in range(4):
        v=M[i,j]; ax.text(j,i,f'{v:.0f} %',ha='center',va='center',fontsize=9.5,color='white' if v>55 else INK2)
for b in sizes[:-1]: ax.axhline(b-0.5,color='white',lw=3)
ax.set_title('Share of documents containing each word, by class\n',loc='left',fontsize=13)
fig.text(0.02,0.015,'Up to four most distinctive words per class, computed from the dataset',color=INK2,fontsize=9)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig('docs/distinctive_words.png',dpi=150)
