#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2020-02-24 16:18:12 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trotta@cua.edu>
#
# Copyright (c) trottar
#

from g4epy import Geant4Eic

g4e=Geant4Eic(detector='jleic',beamline='erhic')\
                        .source('../TDIS_lund.dat')\
                        .output('jleic_mesonMC')\
                        .beam_on(1)\
                        .vis()\
                        .run()