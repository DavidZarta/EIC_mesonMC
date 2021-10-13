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
#fpi = df['fpi']
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

qbinarray = [7,15,30,60,120,240 ,480,1000]
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
'''
cut60 = ["Q2cut3","ycut"]
cut120 = ["Q2cut4","ycut"]
cut240 = ["Q2cut5","ycut"]
cut480 = ["Q2cut6","ycut"]
cut1000 = ["Q2cut7","ycut"]
'''
cut60 = ["Q2cut3","ycut","xLcut80"]
cut120 = ["Q2cut4","ycut","xLcut80"]
cut240 = ["Q2cut5","ycut", "xLcut80"]
cut480 = ["Q2cut6","ycut", "xLcut80"]
cut1000 = ["Q2cut7","ycut", "xLcut80"]

cut200= ["Q2cut0","xLcut85","ycut"]
cut222= ["Q2cut1","xLcut85","ycut"]                #added Q^2 7,15,30
cut244= ["Q2cut2","xLcut85","ycut"]
cut267 = ["Q2cut3","xLcut85","ycut"]
cut289 = ["Q2cut4","xLcut85","ycut"]
cut311 = ["Q2cut5","xLcut85","ycut"]
cut333 = ["Q2cut6","xLcut85","ycut"]
cut356 = ["Q2cut7","xLcut85","ycut"]       #endddd

''''
def F2pi(xpi, Q2):
    points,values=np.load('./../../analysis/interpGrids/xpiQ2.npy'),np.load('./../../analysis/interpGrids/F2pi.npy')
    F2pi=lambda xpi,Q2: griddata(points,values,(np.log10(xpi),np.log10(Q2)))
    return F2pi(xpi,Q2)

# Calculate cross-section using Patrick's interpolate grid
def ds_dxdQ2dxLdt(x, xL,t):
    points60,values60=np.load('./../../analysis/xsec/pointsxsec60.npy'),np.load('./../../analysis/xsec/valuesxsec60.npy')
    points120,values120=np.load('./../../analysis/xsec/pointsxsec120.npy'),np.load('./../../analysis/xsec/valuesxsec120.npy')
    points240,values240=np.load('./../../analysis/xsec/pointsxsec240.npy'),np.load('./../../analysis/xsec/valuesxsec240.npy')
    points480,values480=np.load('./../../analysis/xsec/pointsxsec480.npy'),np.load('./../../analysis/xsec/valuesxsec480.npy')
    sigma60=lambda x,xL,t: griddata(points60,values60,(x,xL,t))
    sigma120=lambda x,xL,t: griddata(points120,values120,(x,xL,t))
    sigma240=lambda x,xL,t: griddata(points240,values240,(x,xL,t))
    sigma480=lambda x,xL,t: griddata(points480,values480,(x,xL,t))

    return [sigma60(x,xL,t),sigma120(x,xL,t),sigma240(x,xL,t),sigma480(x,xL,t)]
    '''
def F2pi(xpi, Q2):
    points,values=np.load('./analysis/interpGrids/xpiQ2.npy'),np.load('./analysis/interpGrids/F2pi.npy')
    F2pi=lambda xpi,Q2: griddata(points,values,(np.log10(xpi),np.log10(Q2)))
    return F2pi(xpi,Q2)
fpi=F2pi(xpi,Q2)
# Calculate cross-section using Patrick's interpolate grid
def ds_dxdQ2dxLdt(x, xL,t):
    points60,values60=np.load('./analysis/xsec/pointsxsec60.npy'),np.load('./analysis/xsec/valuesxsec60.npy')
    points120,values120=np.load('./analysis/xsec/pointsxsec120.npy'),np.load('./analysis/xsec/valuesxsec120.npy')
    points240,values240=np.load('./analysis/xsec/pointsxsec240.npy'),np.load('./analysis/xsec/valuesxsec240.npy')
    points480,values480=np.load('./analysis/xsec/pointsxsec480.npy'),np.load('./analysis/xsec/valuesxsec480.npy')
    sigma60=lambda x,xL,t: griddata(points60,values60,(x,xL,t))
    sigma120=lambda x,xL,t: griddata(points120,values120,(x,xL,t))
    sigma240=lambda x,xL,t: griddata(points240,values240,(x,xL,t))
    sigma480=lambda x,xL,t: griddata(points480,values480,(x,xL,t))
    return [sigma60(x,xL,t),sigma120(x,xL,t),sigma240(x,xL,t),sigma480(x,xL,t)]
def fpivxpi_Plot2():
    
    f = plt.figure(figsize=(11.69,8.27))
    plt.rcParams.update({'font.size': 15})
    plt.style.use('classic')
    #plt.title("{0} $\leq$ xL $\leq$ {1}".format(xlmin,xlmax))
    
    ax = f.add_subplot(221)
    xpiscat4 = ax.scatter(cut.applyCuts(xpi,cut60),cut.applyCuts(f2pi,cut60),color="y",label="HERA",s=150,marker="x")#hera
    plt.plot([.02,.08,.5],[.28,.22, .16],color="r",label="GRV fit")
    ax.text(0.25, 0.65, '$Q^2$=60 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    xpiscat4n = ax.errorbar(cut.applyCuts(xpi,cut60),cut.applyCuts(fpi,cut60),yerr=np.sqrt(cut.applyCuts(lumi,cut60))/cut.applyCuts(lumi,cut60),fmt='.',label='NLO PDF',ecolor='cyan',capsize=2, capthick=2,marker="o",markersize=10)
    #plt.plot([0.1,0.5,0.95],[.2,0.15,0.13],color="y")
    plt.xscale('log')
    plt.ylim(0,0.3)
    plt.xlim(1e-3,1.)
    plt.legend()
    leg=plt.legend(loc='lower left')
    ax.text(0.25, 0.65, '$Q^2$=60 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_yticks([0.0,0.1,0.2,0.3])
    ax.set_xticks([1e-2,1e-1])
    

    plt.ylabel('$F^{\pi}_{2}$', fontsize=20)
    
    ax = f.add_subplot(222)
    xpiscat5 = ax.scatter(cut.applyCuts(xpi,cut120),cut.applyCuts(f2pi,cut120),color="y",s=150,marker="x")#hera
    plt.plot([0.04,.1,.5],[0.26,.19,0.15], label="GRV fit",color="r")
    ax.text(0.25, 0.65, '$Q^2$=120 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    xpiscat5 = ax.errorbar(cut.applyCuts(xpi,cut120),cut.applyCuts(fpi,cut120),yerr=np.sqrt(cut.applyCuts(lumi,cut120))/cut.applyCuts(lumi,cut120),fmt='.',label='$Q^2$=120 $GeV^2$',ecolor='cyan',capsize=2, capthick=2,marker="o",markersize=10)
    #plt.plot([0.1,0.5,0.7],[0.17,0.15,0.12], label="GRV fit",color="y")
    plt.xscale('log')
    plt.ylim(0,0.3)
    plt.xlim(1e-3,1.)
    ax.text(0.25, 0.65, '$Q^2$=120 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_yticks([0.0,0.1,0.2,0.3])
    ax.set_xticks([1e-2,1e-1])
    
    ax = f.add_subplot(223)
    xpiscat6 = ax.scatter(cut.applyCuts(xpi,cut240),cut.applyCuts(f2pi,cut240),color="y",s=150,marker="x")#hera
    plt.plot([0.05,.09,.9],[0.29,.23,0.13],  label="GRV fit",color="r")
    ax.text(0.25, 0.65, '$Q^2$=240 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    xpiscat6 = ax.errorbar(cut.applyCuts(xpi,cut240),cut.applyCuts(fpi,cut240),yerr=np.sqrt(cut.applyCuts(lumi,cut240))/cut.applyCuts(lumi,cut240),fmt='.',label='$Q^2$=240 $GeV^2$',ecolor='cyan',capsize=2, capthick=2,marker="o",markersize=10)
    #plt.plot([0.1,0.2,0.95],[0.2,0.17,0.12], label="GRV fit",color="y")
    plt.xscale('log')
    plt.ylim(0,0.3)
    plt.xlim(1e-3,1.)
    ax.text(0.25, 0.65, '$Q^2$=240 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    ax.set_yticks([0.0,0.1,0.2,0.3])
    ax.set_xticks([1e-2,1e-1])

    ax = f.add_subplot(224)
    xpiscat7 = ax.scatter(cut.applyCuts(xpi,cut480),cut.applyCuts(f2pi,cut480),color="y",s=150,marker="x")#hera
    plt.plot([0.06,.1,.7],[0.23,.20,0.14],  label="GRV fit",color="r")
    ax.text(0.25, 0.65, '$Q^2$=480 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    xpiscat7 = ax.errorbar(cut.applyCuts(xpi,cut480),cut.applyCuts(fpi,cut480),yerr=np.sqrt(cut.applyCuts(lumi,cut480))/cut.applyCuts(lumi,cut480),fmt='.',label='$Q^2$=480 $GeV^2$',ecolor='cyan',capsize=2, capthick=2,marker="o",markersize=10)
    #plt.plot([0.2,0.5,0.9],[0.15,0.14,0.13], label="GRV fit",color="y")
    plt.xscale('log')
    plt.ylim(0,0.3)
    plt.xlim(1e-3,1.)
    ax.text(0.25, 0.65, '$Q^2$=480 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_yticks([0.0,0.1,0.2,0.3])
    ax.set_xticks([1e-2,1e-1])

    plt.xlabel('$x_\pi$', fontsize=20)    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0,wspace=0.0)

    plt.style.use('default')


def fpivxpi_Plot():
    
    f = plt.figure(figsize=(11.69,8.27))
    #plt.rcParams.update({'font.size': 15})
    plt.style.use('classic')
   # plt.title("{0} $\leq$ xL $\leq$ {1}".format(xlmin,xlmax))

    #good 7 !
    ax = f.add_subplot(331)
    xpiscat2 = ax.scatter(cut.applyCuts(xpi,cut200),cut.applyCuts(f2pi,cut200))
    plt.plot([0.001,0.008,.04],[.29,.2, .13], label="GRV fit",color="r")        #change
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=7 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])                                     #change
    #ax.set_xticks([1e-4,1])

    #good 15 !
    ax = f.add_subplot(332)
    xpiscat2 = ax.scatter(cut.applyCuts(xpi,cut222),cut.applyCuts(f2pi,cut222))
    plt.plot([0.009,0.09,.2],[0.29,.18, .14], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=6=15 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 30 !
    ax = f.add_subplot(333)
    xpiscat3 = ax.scatter(cut.applyCuts(xpi,cut244),cut.applyCuts(f2pi,cut244))
    plt.plot([0.01,0.05,.2],[0.29,.21,.16], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=30 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 60 !
    ax = f.add_subplot(334)
    xpiscat4 = ax.scatter(cut.applyCuts(xpi,cut267),cut.applyCuts(f2pi,cut267))
    plt.plot([.02,.08,.5],[.28,.22, .16], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1.)
    ax.text(0.25, 0.65, '$Q^2$=60 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])
    
    plt.ylabel('$F^{\pi}_{2}$', fontsize=20)

    #good 120 !
    ax = f.add_subplot(335)
    xpiscat5 = ax.scatter(cut.applyCuts(xpi,cut289),cut.applyCuts(f2pi,cut289))
    plt.plot([0.04,.1,.5],[0.26,.19,0.15], label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=120 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 240 !
    ax = f.add_subplot(336)
    xpiscat6 = ax.scatter(cut.applyCuts(xpi,cut311),cut.applyCuts(f2pi,cut311))
    plt.plot([0.05,.09,.9],[0.29,.23,0.13],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=240 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6]) 
    #ax.set_xticks([1e-4,1])

    #good 480
    ax = f.add_subplot(337)
    xpiscat7 = ax.scatter(cut.applyCuts(xpi,cut333),cut.applyCuts(f2pi,cut333))
    plt.plot([0.06,.1,.7],[0.23,.20,0.14],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=480 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    #good 1000
    ax = f.add_subplot(338)
    xpiscat8 = ax.scatter(cut.applyCuts(xpi,cut356),cut.applyCuts(f2pi,cut356))
    plt.plot([0.1,.5,1],[0.21,.16,0.13],  label="GRV fit",color="r")
    plt.xscale('log')
    plt.ylim(0.12,0.3)                                                          #change
    plt.xlim(.0001,1)
    ax.text(0.25, 0.65, '$Q^2$=1000 $GeV^2$', transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left')
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_yticks([0,.1,.2,.3,.4,0.5,0.6])
    #ax.set_xticks([1e-4,1])

    plt.ylabel('$F_2$', fontsize=20)                 #thing to put 2 and pi,EF
    plt.xlabel('$x_\pi$', fontsize=20)    
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.0,wspace=0.0)

    plt.style.use('default')


def main() :
    fpivxpi_Plot2()
   # phaseSpace_Plots()
    plt.show()
if __name__=='__main__': main()