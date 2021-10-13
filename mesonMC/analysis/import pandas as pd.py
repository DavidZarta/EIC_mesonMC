import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

import sys
from sys import path
sys.path.insert(1,'./src/process/cuts/') # Note: this is relative to the bash script NOT this python script!
import cuts as c

kinematics = sys.argv[1]

xbinwidth = float(sys.argv[2])
qbinwidth = float(sys.argv[3])
tbinwidth = float(sys.argv[4])
xLbinwidth = float(sys.argv[5])

#xlmin,xlmax = 0.8000,0.8100
xlmin,xlmax = 0.8400,0.8500

df = pd.read_csv(r'./src/process/datafiles/x{0:0.3f}q{1:0.1f}t{2:0.3f}xL{3:0.3f}_{4}.csv'.format(xbinwidth,qbinwidth,tbinwidth,xLbinwidth,kinematics)) # xL bin, no t bin
print(df)

xbj = df['TDIS_xbj']
Q2 = df['TDIS_Q2']
fpi = df['fpi']
t = df['TDIS_t']
xL = df['xL']
y = df['TDIS_y']
sigma_dis = df['sigma_dis']
f2N = df['f2N']
xpi = df['xpi']
#xpi = xbj/(1.-xL)
ypi = df['ypi']
tpi = df['tpi']
lumi = df['tot_int_lumi']

def densityPlot(x,y,title,xlabel,ylabel,binx,biny,
                    xmin=None,xmax=None,ymin=None,ymax=None,cuts=None,fig=None,ax=None,layered=True):

    if ax or fig:
        print("")
    else:
        fig, ax = plt.subplots(tight_layout=True,figsize=(11.69,8.27))

    # norm=colors.LogNorm() makes colorbar normed and logarithmic
    hist = ax.hist2d(x, y,bins=(binx,biny),norm=colors.LogNorm())
    if layered is True :
        plt.colorbar(hist[3], ax=ax, spacing='proportional', label='Number of Events')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    return fig

# Create cut dictionary
cutDict = {}

qbinarray = [7,15,30,60,120,240,480,1000]
#qbinarray = np.arange(qbinwidth/2,1000.,qbinwidth).tolist()
for i,q in enumerate(qbinarray) :
    qtmp = '{"Q2cut%i" : ((%0.1f <= Q2) & (Q2 <= %0.1f))}' % (i,qbinarray[i]-qbinwidth/2,qbinarray[i]+qbinwidth/2)
    print('{"Q2cut%i" : ((%0.1f <= Q2) & (Q2 <= %0.1f))}' % (i,qbinarray[i]-qbinwidth/2,qbinarray[i]+qbinwidth/2))
    cutDict.update(eval(qtmp))

xarray = np.arange(xbinwidth/2,1.0,xbinwidth).tolist()
for i,x in enumerate(xarray):
    xtmp = '{"xcut%i" : ((%0.4f <= xbj) & (xbj <= %0.4f))}' % (i,xarray[i]-xbinwidth/2,xarray[i]+xbinwidth/2)
    print('{"xcut%i" : ((%0.4f <= xbj) & (xbj <= %0.4f))}' % (i,xarray[i]-xbinwidth/2,xarray[i]+xbinwidth/2))
    cutDict.update(eval(xtmp))

tarray = np.arange(tbinwidth/2,1.0,tbinwidth).tolist()
for i,tval in enumerate(tarray):
    ttmp = '{"tcut%i" : ((%0.4f <= -t) & (-t <= %0.4f))}' % (i,tarray[i]-tbinwidth/2,tarray[i]+tbinwidth/2)
    print('{"tcut%i" : ((%0.4f <= -t) & (-t <= %0.4f))}' % (i,tarray[i]-tbinwidth/2,tarray[i]+tbinwidth/2))
    cutDict.update(eval(ttmp))

xLarray = np.arange(xLbinwidth/2,1.0,xLbinwidth).tolist()
for i,x in enumerate(xLarray):
    xLtmp = '{"xLcut%i" : ((%0.4f <= xL) & (xL <= %0.4f))}' % (i,xLarray[i]-xLbinwidth/2,xLarray[i]+xLbinwidth/2)
    print('{"xLcut%i" : ((%0.4f <= xL) & (xL <= %0.4f))}' % (i,xLarray[i]-xLbinwidth/2,xLarray[i]+xLbinwidth/2))
    cutDict.update(eval(xLtmp))

ytmp = '{"ycut" : ((0.01 <= y) & (y <= 0.95))}'
cutDict.update(eval(ytmp))
cut = c.pyPlot(cutDict)    

ycut1 = ["ycut"]
