#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2020-04-12 15:33:17 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trotta@cua.edu>
#
# Copyright (c) trottar
#
import ROOT, math, sys
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from ROOT import Math, TLorentzVector, TFile
from array import array

sys.path.insert(0,'/home/trottar/bin/python/')
import root2py as r2p

kinematics = "pi_n_18on275"
# kinematics = "pi_n_5on100"
# kinematics = "k_lambda_5on100"
# kinematics = "k_lambda_18on275"
# kinematics = "pi_p_18on275"

num = 1

rootName = '../OUTPUTS/%s.root' % kinematics
tree = up.open(rootName)["Evnts"]
branch = r2p.pyBranch(tree)

# pdf = matplotlib.backends.backend_pdf.PdfPages("MC_%s_PIDcut.pdf" % kinematics)
# pdf = matplotlib.backends.backend_pdf.PdfPages("MC_%s_Q2_%i.pdf" % (kinematics,num))
# pdf = matplotlib.backends.backend_pdf.PdfPages("MC_%s_t_0p1_%i_cut.pdf" % (kinematics,num))
# pdf = matplotlib.backends.backend_pdf.PdfPages("MC_%s_t_spec_0p01_cut.pdf" % (kinematics))
pdf = matplotlib.backends.backend_pdf.PdfPages("MC_%s_Econs_cut.pdf" % (kinematics)) 

Q2 = branch.findBranch("invts","Q2")
# t = -branch.findBranch("invts","tPrime")
t = -branch.findBranch("invts","tSpectator")
TDIS_znq = tree.array("TDIS_znq")
xpi = tree.array("xpi")
EprE_Lab = tree.array("EprE_Lab")
EpiE_Lab = tree.array("EpiE_Lab")

b = r2p.pyBin()
c = r2p.pyPlot(None)

myfile = TFile(rootName)
mytree = myfile.Get("Evnts")

print(mytree)

scat_electron_theta = np.array([])
pion_theta = np.array([])
neutron_theta = np.array([])

scat_electron_mom = np.array([])
pion_mom = np.array([])
neutron_mom = np.array([])
photon_mom = np.array([])

Q2_cut = np.array([])
t_cut = np.array([])
TDIS_znq_cut = np.array([])

for entryNum in range(0,mytree.GetEntries()):
    mytree.GetEntry(entryNum)
    # lepton0 = TLorentzVector()
    # lepton1 = TLorentzVector()
    # virtual = TLorentzVector()
    photon = getattr(mytree ,"q_Vir.")
    scat_electron = getattr(mytree ,"e_Scat.")
    pion = getattr(mytree ,"pi.")
    neutron = getattr(mytree ,"p1_Sp.")
    
    # if Q2[entryNum] > num:
    # if t[entryNum]<-0.0 and t[entryNum]>-num:
    
    # if EpiE_Lab[entryNum] == pion.E():
    pion_theta = np.append(pion_theta,pion.Theta()*(180/math.pi))
    # pion_mom = np.append(pion_mom,pion.E())
    pion_mom = np.append(pion_mom,EpiE_Lab[entryNum])
    
    # if EprE_Lab[entryNum] == neutron.E():
    neutron_theta = np.append(neutron_theta,neutron.Theta()*(180/math.pi))
    # neutron_mom = np.append(neutron_mom,neutron.E())
    neutron_mom = np.append(neutron_mom,EprE_Lab[entryNum])
    
    scat_electron_theta = np.append(scat_electron_theta,scat_electron.Theta()*(180/math.pi))
    scat_electron_mom = np.append(scat_electron_mom,scat_electron.E())
    photon_mom = np.append(photon_mom,photon.E())
    Q2_cut = np.append(Q2_cut,Q2[entryNum])
    t_cut = np.append(t_cut,t[entryNum])
    TDIS_znq_cut = np.append(TDIS_znq_cut,TDIS_znq[entryNum])

f = plt.figure(figsize=(11.69,8.27))
plt.style.use('default')

ax = f.add_subplot(211)
hist1 = ax.hist(pion_theta,bins=b.setbin(pion_theta,200),label='pi cut',histtype='step', alpha=0.5, stacked=True, fill=True)
hist1a = ax.hist(scat_electron_theta,bins=b.setbin(scat_electron_theta,200),label='e cut',histtype='step', alpha=0.5, stacked=True, fill=True)
hist1b = ax.hist(neutron_theta,bins=b.setbin(neutron_theta,200),label='n cut',histtype='step', alpha=0.5, stacked=True, fill=True)
ax.legend(loc=1)
plt.title('theta', fontsize =20)

ax = f.add_subplot(212)
hist1 = ax.hist(pion_mom,bins=b.setbin(pion_mom,200),label='pi cut',histtype='step', alpha=0.5, stacked=True, fill=True)
hist1a = ax.hist(scat_electron_mom,bins=b.setbin(scat_electron_mom,200),label='e cut',histtype='step', alpha=0.5, stacked=True, fill=True)
hist1b = ax.hist(neutron_mom,bins=b.setbin(neutron_mom,200),label='n cut',histtype='step', alpha=0.5, stacked=True, fill=True)
ax.legend(loc=1)
plt.title('tot_mom', fontsize =20)

# gamma = c.densityPlot(photon_mom,TDIS_znq_cut, 'TDIS_znq vs $\gamma$ tot_mom','tot_mom','TDIS_znq', 200, 200,  b)
# # plt.ylim(-180.,180.)
# # plt.xlim(0.,50.)
# plt.xlabel('tot_mom (GeV)', fontsize =20)
# plt.ylabel('TDIS_znq', fontsize =20)
# plt.title('TDIS_znq vs $\gamma$ tot_mom', fontsize =20)

# emomQ2 = c.densityPlot(scat_electron_mom,Q2_cut, '$Q^2$ vs e tot_mom','tot_mom','$Q^2$', 200, 200,  b)
# # plt.ylim(-180.,180.)
# # plt.xlim(0.,50.)
# plt.xlabel('tot_mom (GeV)', fontsize =20)
# plt.ylabel('$Q^2$ ($GeV^2$)', fontsize =20)
# plt.title('$Q^2$ vs e tot_mom', fontsize =20)

# ethetaQ2 = c.densityPlot(scat_electron_theta,Q2_cut, '$Q^2$ vs e $\Theta$','$\Theta$','$Q^2$', 200, 200,  b)
# # plt.ylim(-180.,180.)
# # plt.xlim(0.,50.)
# plt.xlabel('$\Theta$', fontsize =20)
# plt.ylabel('$Q^2$ ($GeV^2$)', fontsize =20)
# plt.title('$Q^2$ vs e $\Theta$', fontsize =20)

# emomt = c.densityPlot(pion_mom,t_cut, 't vs k tot_mom','tot_mom','t', 200, 200,  b)
# # plt.ylim(-180.,180.)
# # plt.xlim(0.,50.)
# plt.xlabel('tot_mom (GeV)', fontsize =20)
# plt.ylabel('t ($GeV^2$)', fontsize =20)
# plt.title('t vs k tot_mom', fontsize =20)

# ethetat = c.densityPlot(pion_theta,t_cut, 't vs k $\Theta$','$\Theta$','t', 200, 200,  b)
# # plt.ylim(-180.,180.)
# # plt.xlim(0.,50.)
# plt.xlabel('$\Theta$', fontsize =20)
# plt.ylabel('t ($GeV^2$)', fontsize =20)
# plt.title('t vs k $\Theta$', fontsize =20)

phaseSpace = c.densityPlot(scat_electron_mom, scat_electron_theta, 'dir_z vs tot_mom','tot_mom','dir_z', 200, 200,  b)
# plt.ylim(-180.,180.)
# plt.xlim(0.,50.)
plt.xlabel('tot_mom (GeV)', fontsize =20)
plt.ylabel('Theta (deg)', fontsize =20)
plt.title('e cut', fontsize =20)
phaseSpace = c.densityPlot(pion_mom, pion_theta, 'dir_z vs tot_mom','tot_mom','dir_z', 200, 200,  b)
plt.ylim(1.4,1.5)
plt.xlim(40.,50)
plt.xlabel('tot_mom (GeV)', fontsize =20)
plt.ylabel('Theta (deg)', fontsize =20)
plt.title('pi cut', fontsize =20)
phaseSpace = c.densityPlot(neutron_mom, neutron_theta, 'dir_z vs tot_mom','tot_mom','dir_z', 200, 200,  b)
# plt.ylim(-180.,180.)
plt.xlim(250.,300.)
plt.xlabel('tot_mom (GeV)', fontsize =20)
plt.ylabel('Theta (deg)', fontsize =20)
# plt.title('$\Lambda$ cut', fontsize =20)
plt.title('n cut', fontsize =20)

for f in range(1, plt.figure().number):
    pdf.savefig(f)
pdf.close()

plt.show()