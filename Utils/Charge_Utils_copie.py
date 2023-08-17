import sys,os
import tarfile
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import SDFC.link as sdl
import NSSEA as ns
import NSSEA.plot as nsp
import NSSEA.models as nsm
import scipy.stats as sc
import statsmodels.api as sm
import cftime
import seaborn as sns
import time as tm
import pygam as pg
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

from datetime import datetime
import scipy.linalg as scl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import matplotlib.patches as mplpatch
from matplotlib.gridspec import GridSpec

from scipy.stats._morestats import _calc_uniform_order_statistic_medians
import matplotlib.ticker as ticker
from statsmodels.graphics.tsaplots import plot_acf
from tabulate import tabulate

import arviz #for ess univariate, install with conda install -c conda-forge arviz


#Personnal ADD
sys.path.append(os.path.abspath("/home/barbauxo/Documents/Doctorat/03_Travail/2023_05 Calcul Multimodel/Propre/Utils"))

from multiESS import *
from Utils import *
from Transitions import *
from Preparation import *
from MCMC_Run import *
from EQR import *
from Figures import *
from Full_1Run_Analysis import *
from Comp_Functions import *
