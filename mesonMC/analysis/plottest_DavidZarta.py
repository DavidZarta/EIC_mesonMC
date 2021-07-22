#By David Zarta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
#import ROOT
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
f2pi= 0.361*f2N
# Create cut dictionary
cutDict = {}

qbinarray = [225,250,275,300,325,350 ,375,400]
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
 #changes from here on
cut_x2_q0=["xcut2","Q2cut0","ycut"] #Q2=7 GeV^2
cut_x4_q0 = ["xcut4","Q2cut0","ycut"] # Q2= 7 GeV^2
cut_x6_q0= ["xcut6","Q2cut0","ycut"] # Q2= 7 GeV^2
cut_x8_q0 = ["xcut8","Q2cut0","ycut"] # Q2= 7 GeV^2

cut_x2_q1=["xcut2","Q2cut1","ycut"] #Q2=15 GeV^2
cut_x4_q1 = ["xcut4","Q2cut1","ycut"] # Q2= 15 GeV^2    #added Q^2 7,15,30
cut_x6_q1= ["xcut6","Q2cut1","ycut"] # Q2= 15 GeV^2
cut_x8_q1 = ["xcut8","Q2cut1","ycut"] # Q2= 15 GeV^2

cut_x2_q2=["xcut2","Q2cut2","ycut"] #Q2=30 GeV^2
cut_x4_q2 = ["xcut4","Q2cut2","ycut"] # Q2= 30 GeV^2
cut_x6_q2= ["xcut6","Q2cut2","ycut"] # Q2= 30 GeV^2
cut_x8_q2 = ["xcut8","Q2cut2","ycut"] # Q2= 30 GeV^2

cut_q = ["xLcut90","xcut1","ycut"] # Q2= 60 GeV^2       #?

cut_x2_q3 = ["xcut2","Q2cut3","ycut"] # Q2= 60 GeV^2
cut_x4_q3 = ["xcut4","Q2cut3","ycut"] # Q2= 60 GeV^2
cut_x6_q3= ["xcut6","Q2cut3","ycut"] # Q2= 60 GeV^2
cut_x8_q3 = ["xcut8","Q2cut3","ycut"] # Q2= 60 GeV^2

cut_x2_q4 = ["xcut2","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x4_q4 = ["xcut4","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x6_q4= ["xcut6","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x8_q4 = ["xcut8","Q2cut4","ycut"] # Q2= 120 GeV^2

cut_x2_q5 = ["xcut2","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x4_q5 = ["xcut4","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x6_q5= ["xcut6","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x8_q5 = ["xcut8","Q2cut5","ycut"] # Q2= 240 GeV^2

cut_x2_q6 = ["xcut2","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x4_q6 = ["xcut4","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x6_q6= ["xcut6","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x8_q6 = ["xcut8","Q2cut6","ycut"] # Q2= 480 GeV^2

cut_x2_q7 = ["xcut2","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x4_q7 = ["xcut4","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x6_q7= ["xcut6","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x8_q7 = ["xcut8","Q2cut7","ycut"] # Q2= 1000 GeV^2

'''
cut_x2_q4 = ["xcut2","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x4_q4 = ["xcut4","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x6_q4= ["xcut6","Q2cut4","ycut"] # Q2= 120 GeV^2
cut_x8_q4 = ["xcut8","Q2cut4","ycut"] # Q2= 120 GeV^2

cut_x2_q5 = ["xcut2","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x4_q5 = ["xcut4","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x6_q5= ["xcut6","Q2cut5","ycut"] # Q2= 240 GeV^2
cut_x8_q5 = ["xcut8","Q2cut5","ycut"] # Q2= 240 GeV^2

cut_x2_q6 = ["xcut2","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x4_q6 = ["xcut4","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x6_q6= ["xcut6","Q2cut6","ycut"] # Q2= 480 GeV^2
cut_x8_q6 = ["xcut8","Q2cut6","ycut"] # Q2= 480 GeV^2

cut_x2_q7 = ["xcut2","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x4_q7 = ["xcut4","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x6_q7= ["xcut6","Q2cut7","ycut"] # Q2= 1000 GeV^2
cut_x8_q7 = ["xcut8","Q2cut7","ycut"] # Q2= 1000 GeV^2
'''

                   #this are the Q^2 ingraph. lines 122-1

cut200= ["Q2cut0","xLcut85","ycut"]
cut222= ["Q2cut1","xLcut85","ycut"]                #added Q^2 7,15,30
cut244= ["Q2cut2","xLcut85","ycut"]
cut267 = ["Q2cut3","xLcut85","ycut"]
cut289 = ["Q2cut4","xLcut85","ycut"]
cut311 = ["Q2cut5","xLcut85","ycut"]
cut333 = ["Q2cut6","xLcut85","ycut"]
cut356 = ["Q2cut7","xLcut85","ycut"]       #endddd


def F2pi(xpi, Q2):
    points,values=np.load('./../../analysis/interpGrids/xpiQ2.npy'),np.load('./../../analysis/interpGrids/F2pi.npy')
    F2pi=lambda xpi,Q2: griddata(points,values,(np.log10(xpi),np.log10(Q2)))
    return F2pi(xpi,Q2)

# Calculate cross-section using Patrick's interpolate grid
def ds_dxdQ2dxLdt(x, xL,t):
    points200,values200=np.load('./../../analysis/xsec/pointsxsec7.npy'),np.load('./../../analysis/xsec/valuesxsec7.npy')
    points222,values222=np.load('./../../analysis/xsec/pointsxsec15.npy'),np.load('./../../analysis/xsec/valuesxsec15.npy')
    points244,values244=np.load('./../../analysis/xsec/pointsxsec30.npy'),np.load('./../../analysis/xsec/valuesxsec30.npy')
    points267,values267=np.load('./../../analysis/xsec/pointsxsec60.npy'),np.load('./../../analysis/xsec/valuesxsec60.npy')
    points289,values289=np.load('./../../analysis/xsec/pointsxsec120.npy'),np.load('./../../analysis/xsec/valuesxsec120.npy')
    points311,values311=np.load('./../../analysis/xsec/pointsxsec240.npy'),np.load('./../../analysis/xsec/valuesxsec240.npy')
    points333,values333=np.load('./../../analysis/xsec/pointsxsec480.npy'),np.load('./../../analysis/xsec/valuesxsec480.npy')
    points356,values356=np.load('./../../analysis/xsec/pointsxsec1000.npy'),np.load('./../../analysis/xsec/valuesxsec1000.npy')
    sigma200=lambda x,xL,t: griddata(points200,values200,(x,xL,t))
    sigma222=lambda x,xL,t: griddata(points222,values222,(x,xL,t))
    sigma244=lambda x,xL,t: griddata(points244,values244,(x,xL,t))
    sigma267=lambda x,xL,t: griddata(points267,values267,(x,xL,t))
    sigma289=lambda x,xL,t: griddata(points289,values289,(x,xL,t))
    sigma311=lambda x,xL,t: griddata(points311,values311,(x,xL,t))
    sigma333=lambda x,xL,t: griddata(points333,values333,(x,xL,t))
    sigma356=lambda x,xL,t: griddata(points356,values356,(x,xL,t))

    return [sigma200(x,xL,t),sigma222(x,xL,t),sigma244(x,xL,t),sigma267(x,xL,t),sigma289(x,xL,t),sigma311(x,xL,t),sigma333(x,xL,t),sigma356(x,xL,t)]

def fpivxpi_Plot():
    
    f = plt.figure(figsize=(11.69,8.27))
    #plt.rcParams.update({'font.size': 15})
    plt.style.use('classic')
   # plt.title("{0} $\leq$ xL $\leq$ {1}".format(xlmin,xlmax))

    #good 7 !
    ax = f.add_subplot(331)
    xpiscat2 = ax.scatter(cut.applyCuts(xpi,cut200),cut.applyCuts(f2pi,cut200))
    plt.plot([0.2,0.5,.9],[.15,.145, .139], label="GRV fit",color="r")        #change
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=225 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])                                     #change
    #ax.set_xticks([1e-4,1])

    #good 15 !
    ax = f.add_subplot(332)
    xpiscat2 = ax.scatter(cut.applyCuts(xpi,cut222),cut.applyCuts(f2pi,cut222))
    plt.plot([0.2,0.5,.9],[0.167,.155, .14], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=250 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 30 !
    ax = f.add_subplot(333)
    xpiscat3 = ax.scatter(cut.applyCuts(xpi,cut244),cut.applyCuts(f2pi,cut244))
    plt.plot([0.3,0.65,.95],[0.166,.15,.14], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=275 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 60 !
    ax = f.add_subplot(334)
    xpiscat4 = ax.scatter(cut.applyCuts(xpi,cut267),cut.applyCuts(f2pi,cut267))
    plt.plot([.5,.75,.95],[.155,.14, .135], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=300 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])
    
    plt.ylabel('$F^{\pi}_{2}$', fontsize=20)

    #good 120 !
    ax = f.add_subplot(335)
    xpiscat5 = ax.scatter(cut.applyCuts(xpi,cut289),cut.applyCuts(f2pi,cut289))
    plt.plot([0.6,.75,1.5],[0.145,.13,0.12], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=325 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 240 !
    ax = f.add_subplot(336)
    xpiscat6 = ax.scatter(cut.applyCuts(xpi,cut311),cut.applyCuts(f2pi,cut311))
    plt.plot([0.6,.75,1.5],[0.14,.13,0.12],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=350 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6]) 
    #ax.set_xticks([1e-4,1])

    #good 480
    ax = f.add_subplot(337)
    xpiscat7 = ax.scatter(cut.applyCuts(xpi,cut333),cut.applyCuts(f2pi,cut333))
    plt.plot([0.7,.75,1.5],[0.139,.13,0.12],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=375 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 1000
    ax = f.add_subplot(338)
    xpiscat8 = ax.scatter(cut.applyCuts(xpi,cut356),cut.applyCuts(f2pi,cut356))
    plt.plot([0.75,.85,1.5],[0.137,.13,0.12],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=400 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    plt.ylabel('$F_2$', fontsize=20)                 #thing to put 2 and pi,EF
    plt.xlabel('$x_\pi$', fontsize=20)    
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.0,wspace=0.0)

    plt.style.use('default')


def main() :
    fpivxpi_Plot()
   # phaseSpace_Plots()
    plt.show()
if __name__=='__main__': main()