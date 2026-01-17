#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 15:13:35 2026

@author: andrew
"""

import numpy as np
import os
import time
from os.path import expanduser
from datetime import date
from operator import mul
from functools import reduce
from math import sin, cos, pi
from IPython.display import clear_output
import IscaOpt
    
home = expanduser("~")
current = os.getcwd()

import warnings
warnings.filterwarnings("ignore")


def DTLZ2_Function(decision_vector):
    
    """DTLZ2 multiobjective function. It returns a tuple of *obj* values. 
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.
    
    :math:`g(\\mathbf{x}_m) = \\sum_{x_i \in \\mathbf{x}_m} (x_i - 0.5)^2`
    :math:`f_{\\text{DTLZ2}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i\pi)`
    :math:`f_{\\text{DTLZ2}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\pi)`
    :math:`\\ldots`
    :math:`f_{\\text{DTLZ2}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}\pi )`
    
    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    obj=2
    xc = decision_vector[:obj-1]
    xm = decision_vector[obj-1:]
    g = sum((xi-0.5)**2 for xi in xm)
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(obj-2, -1, -1))

    return f    


if __name__ == '__main__':
  
    # Start time
    start_sim = time.time()

    # sim_id
    today = str(date.today())
    sim_id = today
    
    # initial samples
    sim_file = 'initial_samples_0.npz'        # put the columns in the right order
    
    # reference vector
    reference_vector = np.array([0.9, 0.9])
    
    # kernel
    kernel_name = 'Matern52'
    
    # budget
    number_of_LHS_samples = 43 
    number_of_BO_samples = 100
    budget = number_of_LHS_samples + number_of_BO_samples
    
    # number of dimensions
    n_dim=4
    
    # upper and lower bounds
    lb=np.zeros(n_dim)
    ub=np.ones(n_dim)
    
    # settings
    settings = {\
        'n_dim': 4,\
        'n_obj': 2,\
        'run': sim_id,\
        'lb': lb,\
        'ub': ub,\
        'ref_vector': reference_vector,\
        'method_name': 'MPoI',\
        'budget':budget,\
        'n_samples':number_of_LHS_samples,\
        'visualise':False,\
        'multisurrogate':True,\
        'init_file':sim_file, \
        'kern':kernel_name}
        
    # optimise
    res = IscaOpt.Optimiser.EMO(func=DTLZ2_Function, fargs=(), fkwargs={}, \
                                cfunc=None, cargs=(), ckwargs={}, \
                                settings=settings)
    clear_output()
