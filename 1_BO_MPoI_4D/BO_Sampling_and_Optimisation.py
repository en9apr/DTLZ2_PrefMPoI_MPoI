#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:31:00 2019

@author: andrew
"""

import numpy as np
import os
import subprocess
import time
from scipy.stats import beta as B
from pyDOE import lhs as LHS
from matplotlib.tri import Triangulation
from stl import mesh
from stl import stl
from scipy import spatial
from multiprocessing import Process, Queue, current_process, freeze_support
from os.path import expanduser
import shutil
from datetime import datetime
#import IscaOpt
from datetime import date
from scipy import optimize
from random import randint
from random import seed as seeding

home = expanduser("~")
current = os.getcwd()

# APR added imports
from math import sin, cos, pi, sqrt
from operator import mul
from functools import reduce
# APR added imports



import warnings
warnings.filterwarnings("ignore")





#############
# Environment
############# 
# On ISCA, there can be batches of up to the maximum number of samples

#environment = 'isambard'
#environment = 'isambard_test'
environment = 'isca'
#environment = 'isca_test_mesh'
#environment = 'local'





#environment = 'isca_test_mesh'


if (environment == 'local' or environment == 'isca_test' or environment == 'isambard_test' or environment == 'isca_test_mesh'):
    no_of_nodes = 10

else:    
    no_of_nodes = 10

##########
# Sampling
##########
#sampling = 'manual'
sampling = 'latin'
#sampling = 'single'
#sampling ='double'


#########
# Process
#########
process = 'BO'
#process = 'LHS'
#process = 'BO2'

##########################
# Check the plotting only?
##########################
plotting_only = False

##########################
# Selection of bash script
##########################

# this bash script is for the OpenFOAM run
if environment == 'isca':    
    bash = 'isca_parallel.sh'
elif environment == 'isca_test':
    bash = 'isca_parallel_test.sh' 
elif environment == 'isca_test_mesh':
    bash = 'isca_mesh_run.sh'
elif environment == 'isambard':
    bash = 'isambard_parallel.sh'
elif environment == 'isambard_test':
    bash = 'isambard_parallel_test.sh' 
else:
    bash = 'local_parallel.sh'

if process == 'LHS':
    if environment == 'isambard':
        bash = 'isambard_parallel_lhs.sh'
###############################################################################
# GEOMETRIC FUNCTIONS
###############################################################################

#########################################################
# Trays class used by BOTH INITIAL_SAMPLING AND OPTIMISER
#########################################################

class Trays():

    def __init__(self, ub, lb):
        
        self.ub = ub
        self.lb = lb        
        
        self.separation = 22.0
        #self.blockage = 19.0
#        self.y_top = 439.5 # top of tray
#        self.y_bottom = 101.0 # bottom of tray
        self.y_top = 503.622 # top of tray
        self.y_bottom = 165.122 # bottom of tray
        self.benching = 367.5
        self.lip = 401.622-5.7
        
        theta_linear = np.linspace(0, 2 * np.pi, 125)
        
        #################################################
        # geometric progression in y: benching constraint
        #################################################
        no_of_layers_to_middle = 25
        expansion_factor = 1.05
        thickness_multiplier = 0.5
        
        first_layer_thickness=(thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers_to_middle))
        
        thickness=np.zeros(no_of_layers_to_middle*2 +1)
        
        thickness[1]=first_layer_thickness
        thickness[no_of_layers_to_middle*2]=first_layer_thickness
        
        for layer in range(2,no_of_layers_to_middle+1):
            thickness[layer] = expansion_factor*thickness[layer-1]
        
        for layer in range(no_of_layers_to_middle*2-1, no_of_layers_to_middle, -1):
            thickness[layer] = expansion_factor*thickness[layer+1]
        
        z_normal=np.zeros((no_of_layers_to_middle*2)+1)
        
        for layer in range(1, (no_of_layers_to_middle*2)+1):
            z_normal[layer] = z_normal[layer-1] + thickness[layer]
            
        y_linear = z_normal
        y_grid, theta_grid = np.meshgrid(y_linear, theta_linear)
        z_global = np.ravel(y_grid)   

        four_two_five = np.zeros((z_global.shape[0],3))

        self.offset_large = np.copy(four_two_five) # offset 2
    
        for j in range(len(self.offset_large)):
            self.offset_large[j,2] -= 64.122  
            
            
        ############################################
        # geometric progression in y: lip constraint
        ############################################
        no_of_layers = 25
        expansion_factor = 1.05
        thickness_multiplier = 1
        
        first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers))
        
        thickness = np.zeros(no_of_layers +1)
        
        thickness[no_of_layers] = first_layer_thickness
        
        for layer in range(no_of_layers-1, 0, -1):
            thickness[layer] = expansion_factor*thickness[layer+1]
        
        z_normal = np.zeros((no_of_layers)+1)
        
        for layer in range(1, (no_of_layers)+1):
            z_normal[layer] = z_normal[layer-1] + thickness[layer]
            
        y_linear = z_normal
        y_grid, theta_grid = np.meshgrid(y_linear, theta_linear)
        z_global = np.ravel(y_grid)   

        four_two_five = np.zeros((z_global.shape[0],3))            
            
        self.offset_small = np.copy(four_two_five)

        for i in range(len(self.offset_small)):
            self.offset_small[i,2] -= 35.7

###########
# APR ADDED
###########
def stl_lip_constraint(y_bottom, x):
    
    x_bottom=0
    angle_bottom=0
    a_bottom = 50.0
    b_bottom = 50.0
    
    ###################
    # Values on the lip
    ###################
    # angular resolution is twice grid resolution - angle: resolution in Pointwise 47 * 4 = 188. Choose 400
    # 100 is temporary low resolution
    theta_linear_lip = np.linspace(0, 2 * np.pi, 125)

    ############################
    # geometric progression in y
    ############################        
    no_of_layers = 25 # was 87
    expansion_factor = 1.05
    thickness_multiplier = 1
    
    first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers))
    
    thickness = np.zeros(no_of_layers +1)
    
    #thickness[1]=first_layer_thickness
    thickness[no_of_layers] = first_layer_thickness
    
    for layer in range(no_of_layers-1, 0, -1):
        thickness[layer] = expansion_factor*thickness[layer+1]
    
    z_normal = np.zeros((no_of_layers)+1)
    
    for layer in range(1, (no_of_layers)+1):
        z_normal[layer] = z_normal[layer-1] + thickness[layer]
        
    # Use tray y in order to compute lip values
    z_bottom = 165.122-30
    z_top = 503.622-30 # override the input and give a constant
    z_bottom = z_top-72
    z_linear_lip = z_normal*(z_top-z_bottom) + z_bottom   
    
    # grid
    z_grid_lip, theta_grid_lip = np.meshgrid(z_linear_lip, theta_linear_lip)
    
    ############################
    # geometric progression in y
    ############################      
    
    # ellipse radii
    a_linear_lip = evaluate((z_linear_lip-z_top)/(222.122-560.622), 440.0, a_bottom, x)
    b_linear_lip = evaluate((z_linear_lip-z_top)/(222.122-560.622), 440.0, b_bottom, x)
    
#    print("lip",z_normal[0], z_normal[-1])
#    print("lip",z_linear_lip[0], z_linear_lip[-1])
#    print("lip",a_linear_lip[0], a_linear_lip[-1])

    # centre locations c[0] = x location, c[1] = y_location
    x_linear_lip = evaluate((z_linear_lip-z_top)/(222.122-560.622), 0.0, x_bottom, x)
    
    # parameter 2: y-centre location
    y_linear_lip = evaluate((z_linear_lip-z_top)/(222.122-560.622), 0.0, y_bottom, x)
        
    # angle
    angle_linear_lip = (np.pi/180.0)*evaluate((z_linear_lip-z_top)/(222.122-560.622), 0.0, angle_bottom, x)

    # x_surface
    x_surface = x_linear_lip + a_linear_lip * np.cos(theta_grid_lip)
    
    # y_surface
    y_surface = y_linear_lip + b_linear_lip * np.sin(theta_grid_lip)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear_lip)) * (x_surface - x_linear_lip) - (np.sin(angle_linear_lip)) * (y_surface - y_linear_lip) + x_linear_lip)
    y = np.ravel((np.sin(angle_linear_lip)) * (x_surface - x_linear_lip) + (np.cos(angle_linear_lip)) * (y_surface - y_linear_lip) + y_linear_lip)
    z = np.ravel(z_grid_lip)

    return np.array([x,y,z]).T

###########
# APR ADDED
###########
def stl_benching_constraint(y_bottom, x):
    
    x_bottom=0
    angle_bottom=0
    a_bottom = 50.0
    b_bottom = 50.0
    
    # angular resolution is twice grid resolution - angle: resolution in Pointwise 47 * 4 = 188. Choose 400
    theta_linear = np.linspace(0, 2 * np.pi, 125)
    
    ############################
    # geometric progression in y
    ############################
    no_of_layers_to_middle = 25
    expansion_factor = 1.05
    thickness_multiplier = 0.5
    
    first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers_to_middle))
    
    thickness = np.zeros(no_of_layers_to_middle*2 +1)
    
    thickness[1] = first_layer_thickness
    thickness[no_of_layers_to_middle*2] = first_layer_thickness
    
    for layer in range(2,no_of_layers_to_middle+1):
        thickness[layer] = expansion_factor*thickness[layer-1]
    
    for layer in range(no_of_layers_to_middle*2-1, no_of_layers_to_middle, -1):
        thickness[layer] = expansion_factor*thickness[layer+1]
    
    z_normal=np.zeros((no_of_layers_to_middle*2)+1)
    
    for layer in range(1, (no_of_layers_to_middle*2)+1):
        z_normal[layer] = z_normal[layer-1] + thickness[layer]

    ############################
    # geometric progression in y
    ############################

    z_top_tray = 367.5

    z_bottom_tray = 101.0 # override the input and give a constant
    z_linear_tray = z_normal*(z_top_tray-z_bottom_tray) + z_bottom_tray
    
    # grid
    z_grid, theta_grid = np.meshgrid(z_linear_tray, theta_linear)
    
    # ellipse radii
    a_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 440.0, a_bottom, x)
    b_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 440.0, b_bottom, x)
    
    # centre locations c[0] = x location, c[1] = z_location
    x_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, x_bottom, x)
    
    # parameter 2: y-centre location
    y_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, y_bottom, x)
        
    # angle
    angle_linear = (np.pi/180.0)*evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, angle_bottom, x)
        
    # x_surface
    x_surface = x_linear + a_linear * np.cos(theta_grid)
    
    # y_surface
    y_surface = y_linear + b_linear * np.sin(theta_grid)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear)) * (x_surface - x_linear) - (np.sin(angle_linear)) * (y_surface - y_linear) + x_linear)
    y = np.ravel((np.sin(angle_linear)) * (x_surface - x_linear) + (np.cos(angle_linear)) * (y_surface - y_linear) + y_linear)
    z = np.ravel(z_grid)

    return np.array([x,y,z]).T





                
##############################################################
# gap_constraint(x, layout) FUNCTION USED BY OPTIMISER AND LHS
##############################################################

def gap_constraint(x, trays):
    """
    Checks only the gap constraint: writes no constraints.npz file.

    Parameters.
    -----------
    x (numpy array): the decision vector. 
    layout (Ellipse object): the object for the generation of the pointwise files. 

    Returns whether the constraint was successfully passed.         
    """

    # THIS IS A COPY OF THE GAP CONSTRAINT FROM gap_and_checkMesh_constraint

    # defaults for success:
    gap_success = True
    
    ####################################################################
    # Gap constraint based on minimum distance between frustum and torus 
    #################################################################### 
    xyz_tray_lip_constraint = stl_lip_constraint(x[0], x)
    
    xyz_tray_benching_constraint = stl_benching_constraint(x[0], x)    
        
    lip_offset = xyz_tray_lip_constraint + trays.offset_small        
    
    tree = spatial.cKDTree(lip_offset)
            
    mindist, minid = tree.query(xyz_tray_lip_constraint, n_jobs=16)
    minimum_lip = min(mindist)
        
    benching_offset = xyz_tray_benching_constraint + trays.offset_large
    
    tree = spatial.cKDTree(benching_offset)
            
    mindist, minid = tree.query(xyz_tray_benching_constraint, n_jobs=16)
    minimum_benching = min(mindist)

    if(minimum_benching < trays.separation or minimum_lip < trays.separation):
        gap_success = False
    elif(minimum_benching > trays.separation+1 or minimum_lip > trays.separation+1):
        gap_success = False
        
    return gap_success

def stl_constraint(trays, y_top, y_bottom, x):
    #############################################################
    # Carbon copy of stl_tray but returns numpy arrays not saving
    #############################################################
    a_bottom = 50.0
    b_bottom = 50.0
    
    evaluate_values = evaluate_constraint((trays.y_linear-y_top)/(y_bottom-y_top), x)
    
    a_linear = evaluate_values*(a_bottom-440.0) + 440.0
    b_linear = evaluate_values*(b_bottom-440.0) + 440.0
    x_linear = evaluate_values*(0.0-0.0) + 0.0
    z_linear = evaluate_values*(x[0]-0.0) + 0.0
    angle_linear = (np.pi/180.0)*evaluate_values*(0.0-0.0) + 0.0
       
    x_surface = x_linear + a_linear * np.cos(trays.theta_grid)
    z_surface = z_linear + b_linear * np.sin(trays.theta_grid)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear)) * (x_surface - x_linear) - (np.sin(angle_linear)) * (z_surface - z_linear) + x_linear)
    y = np.ravel((np.sin(angle_linear)) * (x_surface - x_linear) + (np.cos(angle_linear)) * (z_surface - z_linear) + z_linear)
    
    return np.array([x,y,trays.z_global]).T


def evaluate_constraint(positions, x):

    return B.cdf(positions, np.array([x[1]]), np.array([x[2]]))

def evaluate(positions, lb, ub, x):
    
    alphas=np.array([x[1]])
    betas=np.array([x[2]])
    omegas=np.array([1])

    EXT_BETA_CDF = lambda x, a, b : np.array([B.cdf(x, a[i], b[i]) for i in range(len(a))])
    
    return np.dot(omegas, EXT_BETA_CDF(positions, alphas, betas))*(ub-lb) + lb

def stl_benching(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name, x):
    
    a_bottom = 50.0
    b_bottom = 50.0
    
    angle_bottom = 0.0
    x_bottom = 0.0
    
    dir_name = "_"
    for j in range(len(x)):
        dir_name += "{0:.8f}_".format(x[j])
    
    case_path = '/data/HeadCell/case/'
    
    # angular resolution is twice grid resolution - angle: resolution in Pointwise 47 * 4 = 188. Choose 400
    theta_linear = np.linspace(0, 2 * np.pi, 400)
    
    ############################
    # geometric progression in y
    ############################
    no_of_layers_to_middle = 80
    expansion_factor = 1.05
    thickness_multiplier = 0.5
    
    first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers_to_middle))
    
    thickness = np.zeros(no_of_layers_to_middle*2 +1)
    
    thickness[1] = first_layer_thickness
    thickness[no_of_layers_to_middle*2] = first_layer_thickness
    
    for layer in range(2,no_of_layers_to_middle+1):
        thickness[layer] = expansion_factor*thickness[layer-1]
    
    for layer in range(no_of_layers_to_middle*2-1, no_of_layers_to_middle, -1):
        thickness[layer] = expansion_factor*thickness[layer+1]
    
    z_normal=np.zeros((no_of_layers_to_middle*2)+1)
    
    for layer in range(1, (no_of_layers_to_middle*2)+1):
        z_normal[layer] = z_normal[layer-1] + thickness[layer]

    ############################
    # geometric progression in y
    ############################
    f = lambda z: evaluate((z-439.5)/(101.0-439.5), 440.0, a_bottom, x) - 101.0
    
    try:
        root = optimize.newton(f, 101.0)
    except Exception:
        root = 1000000
        pass
    
    z_top_tray = 367.5
    z_bottom_tray = root # override the input and give a constant
    z_linear_tray = z_normal*(z_top_tray-z_bottom_tray) + z_bottom_tray
    
    # grid
    z_grid, theta_grid = np.meshgrid(z_linear_tray, theta_linear)
    
    # ellipse radii
    a_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 440.0, a_bottom, x)
    b_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 440.0, b_bottom, x)
    
    # centre locations c[0] = x location, c[1] = z_location
    x_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, x_bottom, x)
    
    # parameter 2: y-centre location
    y_linear = evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, y_bottom, x)
        
    # angle
    angle_linear = (np.pi/180.0)*evaluate((z_linear_tray-439.5)/(101.0-439.5), 0.0, angle_bottom, x)
        
    # x_surface
    x_surface = x_linear + a_linear * np.cos(theta_grid)
    
    # y_surface
    y_surface = y_linear + b_linear * np.sin(theta_grid)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear)) * (x_surface - x_linear) - (np.sin(angle_linear)) * (y_surface - y_linear) + x_linear)
    y = np.ravel((np.sin(angle_linear)) * (x_surface - x_linear) + (np.cos(angle_linear)) * (y_surface - y_linear) + y_linear)
    z = np.ravel(z_grid)
    
    # triangulation
    tri = Triangulation(np.ravel(z_grid), np.ravel(theta_grid))
    
    ################
    # Save .stl file
    ################
    
    data = np.zeros(len(tri.triangles), dtype=mesh.Mesh.dtype)
    mobius_mesh = mesh.Mesh(data, remove_empty_areas=False)
    mobius_mesh.x[:] = x[tri.triangles]
    mobius_mesh.y[:] = y[tri.triangles]
    mobius_mesh.z[:] = z[tri.triangles]
    
    with open(current + case_path + dir_name + "/" + file_name, 'wb') as output_file:
        #file_name_string = str(file_name)
        mobius_mesh.save(current + case_path + dir_name + "/" + file_name, output_file, mode=stl.ASCII)


def stl_tray(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name, x):
    
    x_bottom = 0.0
    angle_bottom = 0.0
    a_bottom = 50.0
    b_bottom = 50.0
    
    dir_name = "_"
    for j in range(len(x)):
        dir_name += "{0:.8f}_".format(x[j])
    
    case_path = '/data/HeadCell/case/'
    
    # angular resolution is twice grid resolution - angle: resolution in Pointwise 47 * 4 = 188. Choose 400
    theta_linear = np.linspace(0, 2 * np.pi, 400)
    
    ############################
    # geometric progression in y
    ############################
    no_of_layers_to_middle = 80
    expansion_factor = 1.05
    thickness_multiplier = 0.5
    
    first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers_to_middle))
    
    thickness = np.zeros(no_of_layers_to_middle*2 +1)
    
    thickness[1] = first_layer_thickness
    thickness[no_of_layers_to_middle*2] = first_layer_thickness
    
    for layer in range(2,no_of_layers_to_middle+1):
        thickness[layer] = expansion_factor*thickness[layer-1]
    
    for layer in range(no_of_layers_to_middle*2-1, no_of_layers_to_middle, -1):
        thickness[layer] = expansion_factor*thickness[layer+1]
    
    z_normal=np.zeros((no_of_layers_to_middle*2)+1)
    
    for layer in range(1, (no_of_layers_to_middle*2)+1):
        z_normal[layer] = z_normal[layer-1] + thickness[layer]
        
    z_bottom = 165.122
    z_top = 503.622
    z_linear = z_normal*(z_top-z_bottom) + z_bottom
    
    ############################
    # geometric progression in y
    ############################
    
    # grid
    z_grid, theta_grid = np.meshgrid(z_linear, theta_linear)
    
    a_linear = evaluate((z_linear-z_top)/(z_bottom-z_top), 440.0, a_bottom, x)
    b_linear = evaluate((z_linear-z_top)/(z_bottom-z_top), 440.0, b_bottom, x)
    
    x_linear = evaluate((z_linear-z_top)/(z_bottom-z_top), 0.0, x_bottom, x)
    y_linear = evaluate((z_linear-z_top)/(z_bottom-z_top), 0.0, y_bottom, x)
    
    angle_linear = (np.pi/180.0)*evaluate((z_linear-z_top)/(z_bottom-z_top), 0.0, angle_bottom, x)
    
    # x_surface
    x_surface = x_linear + a_linear * np.cos(theta_grid)
    
    # z_surface
    y_surface = y_linear + b_linear * np.sin(theta_grid)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear)) * (x_surface - x_linear) - (np.sin(angle_linear)) * (y_surface - y_linear) + x_linear)
    y = np.ravel((np.sin(angle_linear)) * (x_surface - x_linear) + (np.cos(angle_linear)) * (y_surface - y_linear) + y_linear)
    z = np.ravel(z_grid)
    
    # triangulation
    tri = Triangulation(np.ravel(z_grid), np.ravel(theta_grid))
    
    ################
    # Save .stl file
    ################
    
    data = np.zeros(len(tri.triangles), dtype=mesh.Mesh.dtype)
    mobius_mesh = mesh.Mesh(data, remove_empty_areas=False)
    mobius_mesh.x[:] = x[tri.triangles]
    mobius_mesh.y[:] = y[tri.triangles]
    mobius_mesh.z[:] = z[tri.triangles]
    
    with open(current + case_path + dir_name + "/" + file_name, 'wb') as output_file:
        mobius_mesh.save(current + case_path + dir_name + "/" + file_name, output_file, mode=stl.ASCII)

def stl_lip(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name_lip, x):

    x_bottom = 0.0    
    angle_bottom = 0.0
    a_bottom = 50.0
    b_bottom = 50.0
    
    dir_name = "_"
    for j in range(len(x)):
        dir_name += "{0:.8f}_".format(x[j])
    
    case_path = '/data/HeadCell/case/'
    
    ###################
    # Values on the lip
    ###################
    # angular resolution is twice grid resolution - angle: resolution in Pointwise 47 * 4 = 188. Choose 400
    # 100 is temporary low resolution
    theta_linear_lip = np.linspace(0, 2 * np.pi, 400)

    ############################
    # geometric progression in y
    ############################        
    no_of_layers = 87
    expansion_factor = 1.05
    thickness_multiplier = 1
    
    first_layer_thickness = (thickness_multiplier*(1.0-expansion_factor))/(1.0-(expansion_factor**no_of_layers))
    
    thickness = np.zeros(no_of_layers +1)
    
    #thickness[1]=first_layer_thickness
    thickness[no_of_layers] = first_layer_thickness
    
    for layer in range(no_of_layers-1, 0, -1):
        thickness[layer] = expansion_factor*thickness[layer+1]
    
    z_normal = np.zeros((no_of_layers)+1)
    
    for layer in range(1, (no_of_layers)+1):
        z_normal[layer] = z_normal[layer-1] + thickness[layer]
        
    # Use tray y in order to compute lip values
    z_top_lip = 560.622
    z_bottom_lip = 488.622 # override the input and give a constant
    z_linear_lip = z_normal*(z_top_lip-z_bottom_lip) + z_bottom_lip       
        
    # grid
    z_grid_lip, theta_grid_lip = np.meshgrid(z_linear_lip, theta_linear_lip)
    
    ############################
    # geometric progression in y
    ############################      
    
    # ellipse radii
    a_linear_lip = evaluate((z_linear_lip-560.622)/(222.122-560.622), 440.0, a_bottom, x)
    b_linear_lip = evaluate((z_linear_lip-560.622)/(222.122-560.622), 440.0, b_bottom, x)
    
#    print("lip",z_normal[0], z_normal[-1])
#    print("lip",z_linear_lip[0], z_linear_lip[-1])
#    print("lip",a_linear_lip[0], a_linear_lip[-1])

    # centre locations c[0] = x location, c[1] = y_location
    x_linear_lip = evaluate((z_linear_lip-560.622)/(222.122-560.622), 0.0, x_bottom, x)
    
    # parameter 2: y-centre location
    y_linear_lip = evaluate((z_linear_lip-560.622)/(222.122-560.622), 0.0, y_bottom, x)
        
    # angle
    angle_linear_lip = (np.pi/180.0)*evaluate((z_linear_lip-560.622)/(222.122-560.622), 0.0, angle_bottom, x)

    # x_surface
    x_surface = x_linear_lip + a_linear_lip * np.cos(theta_grid_lip)
    
    # y_surface
    y_surface = y_linear_lip + b_linear_lip * np.sin(theta_grid_lip)
    
    # x, y, z locations
    x = np.ravel((np.cos(angle_linear_lip)) * (x_surface - x_linear_lip) - (np.sin(angle_linear_lip)) * (y_surface - y_linear_lip) + x_linear_lip)
    y = np.ravel((np.sin(angle_linear_lip)) * (x_surface - x_linear_lip) + (np.cos(angle_linear_lip)) * (y_surface - y_linear_lip) + y_linear_lip)
    z = np.ravel(z_grid_lip)
    
    # triangulation
    tri = Triangulation(np.ravel(z_grid_lip), np.ravel(theta_grid_lip))
    
    ################
    # Save .stl file
    ################
    
    data = np.zeros(len(tri.triangles), dtype=mesh.Mesh.dtype)
    mobius_mesh = mesh.Mesh(data, remove_empty_areas=False)
    mobius_mesh.x[:] = x[tri.triangles]
    mobius_mesh.y[:] = y[tri.triangles]
    mobius_mesh.z[:] = z[tri.triangles]
        
    with open(current + case_path + dir_name + "/" + file_name_lip, 'wb') as output_file:
        mobius_mesh.save(current + case_path + dir_name + "/" + file_name_lip, output_file, mode=stl.ASCII)


###############################################################################
## PARALLELISATION FUNCTIONS
###############################################################################

################################################
# queue() function used by INITIAL_SAMPLING ONLY
################################################
    
def queue(filedirs):
    
#    start_sim = time.time()
#    filedirs = initialisation(trays)
#    print('Initialisation wall clock time: ', time.time()-start_sim)

#    if(environment == 'isca_test'):
#
#        subprocess.call(['sh', 'isca_test_dependancy.sh'], \
#              stdout = open('log.bash', 'w+'), \
#              stderr = open('err.bash', 'w+'))
#    
#    elif(environment == 'isambard'):
#        
#        subprocess.call(['sh', 'isambard_dependancy.sh'], \
#              stdout = open('log.bash', 'w+'), \
#              stderr = open('err.bash', 'w+'))        
#        
#        
#        
#        
#    print('Submissions complete')











    task_list = [(mul, (d,)) for d in filedirs]    
    
    # Create queues
    task_queue = Queue()
    done_queue = Queue()
    
    # Submit tasks
    for task in task_list:
        task_queue.put(task)

    # Start worker processes
    for i in range(no_of_nodes):
        Process(target=worker, args=(task_queue, done_queue)).start()    
    
    # Get and print results
    print('Unordered results:')
    for i in range(len(task_list)):
        print('\t', done_queue.get())
    
    # Tell child processes to stop
    for i in range(no_of_nodes):
        task_queue.put('STOP')


##############################################
# mul() function used by INITIAL_SAMPLING ONLY
##############################################
        
# WARNING: completed.txt writing for initial sampling has not been tested        
        
#def mul(d):
#    
#    if ((environment == 'isca') or (environment == 'isca_test_mesh')):
#        start_msub = time.time()
#        now = datetime.now()
#        start_time = now.strftime("%H:%M:%S")
#        subprocess.call(['sbatch', d + '/' + bash], cwd=d, \
#              stdout = open(d + '/log.subprocess', 'w+'), \
#              stderr = open(d + '/err.subprocess', 'w+'))
#
#
#
#        while not os.path.exists(d + '/' + 'completed.txt'):
#            time.sleep(60)
#        now = datetime.now()
#        end_time = now.strftime("%H:%M:%S")
#        
#    elif (environment == 'isca_test'):
#        start_msub = time.time()
#        now = datetime.now()
#        start_time = now.strftime("%H:%M:%S")
#        subprocess.call(['sbatch', d + '/' + bash], cwd=d, \
#              stdout = open(d + '/log.subprocess', 'w+'), \
#              stderr = open(d + '/err.subprocess', 'w+'))
#
#        while not os.path.exists(d + '/' + 'completed.txt'):
#            time.sleep(60)
#        now = datetime.now()
#        end_time = now.strftime("%H:%M:%S")       
#        
#    # submit job using qsub if on isambard:
#    elif (environment == 'isambard_test'):
#        start_msub = time.time()
#        now = datetime.now()
#        start_time = now.strftime("%H:%M:%S")
#        subprocess.call(['qsub', d + '/' + bash], cwd=d, \
#              stdout = open(d + '/log.subprocess', 'w+'), \
#              stderr = open(d + '/err.subprocess', 'w+'))
#
#        while not os.path.exists(d + '/' + 'completed.txt'):
#            time.sleep(60)
#        now = datetime.now()
#        end_time = now.strftime("%H:%M:%S")
#
#    # submit job using qsub if on isambard:
#    elif ((environment == 'isambard') or (environment == 'isambard_one')):
#        start_msub = time.time()
#        now = datetime.now()
#        start_time = now.strftime("%H:%M:%S")
#        subprocess.call(['qsub', d + '/' + bash], cwd=d, \
#              stdout = open(d + '/log.subprocess', 'w+'), \
#              stderr = open(d + '/err.subprocess', 'w+'))
#
#        while not os.path.exists(d + '/' + 'completed.txt'):
#            time.sleep(60)
#        now = datetime.now()
#        end_time = now.strftime("%H:%M:%S")
#    else:
#        start_msub = time.time()
#        now = datetime.now()
#        start_time = now.strftime("%H:%M:%S")
#
#        subprocess.call(['sh', d + '/' + bash], cwd=d, \
#              stdout = open(d + '/log.subprocess', 'w+'), \
#              stderr = open(d + '/err.subprocess', 'w+'))
#        
#        while not os.path.exists(d + '/' + 'completed.txt'):
#            time.sleep(1)
#        now = datetime.now()
#        end_time = now.strftime("%H:%M:%S")
#        
#    return "Start time, " + start_time + " End time, " + end_time + " Elapsed time, " + str(time.time() - start_msub) 

#################################################
# worker() function used by INITIAL_SAMPLING ONLY
#################################################
    
def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

####################################################
# calculate() function used by INITIAL_SAMPLING ONLY
####################################################  
        
def calculate(func, args):

    result = func(*args)
    return '%s says that %s%s = %s' % \
        (current_process().name, func.__name__, args, result)
      
###############################################################################
## SAMPLING FUNCTIONS
###############################################################################

########################################################
# initialisation() FUNCTION USED BY INITAL_SAMPLING ONLY
########################################################
            
def initialisation(trays):
    """
    Samples the design space,
    Writes the decision vector to a file
    Creates an empty initial samples file
    Copies and edits the qsub or msub submission script.

    Parameters.
    -----------
    There are no parameters. 

    Returns a list of directories for submission.         
    """
    # get current directory
    current = os.getcwd()
    
    # set the location of the case path
    case_path = "/data/HeadCell/case/"
    source_path = "/data/HeadCell/source/."
        
    #source_file = 'run_case_initial_sampling.py'
    
    # remove the current case path and create a new case path
    subprocess.call(['rm', '-r', current + case_path])    
    subprocess.call(['mkdir','-p', current + case_path])
    
    
    if(function == 'dtlz2'):
        #n_objectives = 3
        n_dim = 3
        n_samples = 11*n_dim-1
        print('Number of dimensions: ', n_dim)
              
        # APR added
        # read decision vector and write to npz file
        print('Writing empty initial_samples.npz file...')
        initial_X = []
        initial_Y = []
        
        # the name of the sim_file is initial_samples.npz    
        sim_file = 'initial_samples.npz'
        
        # remove npz file if it exists
        subprocess.call(['rm', '-r', current + '/' + sim_file]) 
        
        try:
            np.savez(current + '/' + sim_file, initial_X, initial_Y)
            print('Data saved in file: ', sim_file)
        except Exception as e:
            print(e)
            print('Data saving failed.')
        # APR added        
        
        
        lb=np.zeros(n_dim)
        ub=np.ones(n_dim)
        
        ################################
        # obtain Latin Hypercube samples
        ################################
        print("initialisation(): number of Latin Hypercube samples", str(n_samples))
        samples = lhs_initial_samples(n_dim, ub, lb, n_samples, cfunc=None, cargs=(), ckwargs={})
        print('initialisation(): number of samples that pass constraints, ', len(samples))
        
    
        ###################################################################
        # correct all files using the directory names and create geometries
        ###################################################################
        
        # create filedirs
        filedirs = []
        
        # loop through the list to create directories:
        for s in samples:
            
            # create a working directory from the sample:
            dir_name = "_"
            for j in range(len(s)):
                dir_name += "{0:.8f}_".format(s[j])
            
            # replace any directories containing []
            dir_name = dir_name.replace("[","")
            dir_name = dir_name.replace("]","")
            
            # add the name to a list of directories
            filedirs.append(current + case_path + dir_name)
            
            # create the directory from the last in the list and 
            subprocess.call(['mkdir', filedirs[-1] + '/'])
            
            # copy all source files into the newly created directory
            subprocess.call(['cp', '-r', current + source_path + '/run_case.py', filedirs[-1] + '/run_case.py'])
            subprocess.call(['cp', '-r', current + source_path + '/local_parallel.sh', filedirs[-1] + '/local_parallel.sh'])
            subprocess.call(['cp', '-r', current + source_path + '/isca_parallel_test.sh', filedirs[-1] + '/isca_parallel_test.sh'])
            subprocess.call(['cp', '-r', current + source_path + '/isambard_parallel_test.sh', filedirs[-1] + '/isambard_parallel_test.sh'])
            
            # write the decision vector to a file        
            with open(filedirs[-1] + "/decision_vector.txt", "w+") as myfile:
                for i in range(0,len(s)):
                    myfile.write(str(s[i])+ '\n')
            print("initialisation(): written decision vector to a file")
        
        
            with open(filedirs[-1] + '/run_case.py', 'r') as f:
                data = f.readlines()
            
            # change the line to the correct directory
            for line in range(len(data)):       
                if 'dir_name = None' in data[line]:
                    data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
            
            #write the lines to the file        
            with open(filedirs[-1] + '/run_case.py', 'w+') as f:
                f.writelines(data)
            
            
            # change bash_pointwise in order to run from current directory
            with open(filedirs[-1] + '/local_parallel.sh', 'r') as myfile:
                data_myfile = myfile.readlines()
    
            # read in each line and change the directory location                
            for line in range(len(data_myfile)):       
                if 'cd None' in data_myfile[line]:
                    data_myfile[line] = 'cd ' + current + case_path + dir_name + '\n'       
            
            with open(filedirs[-1] + '/local_parallel.sh', 'w+') as myfile:
                myfile.writelines(data_myfile)
            
            # open the bash script for running
            with open(filedirs[-1] + '/' + 'isca_parallel_test.sh', 'r') as f:
                data = f.readlines()
                
            # change the line to the correct directory
            for line in range(len(data)):       
                if '#SBATCH -D' in data[line]:
                    data[line] = '#SBATCH -D '+ filedirs[-1]+'/' + '\n'
            
            #write the lines to the file        
            with open(filedirs[-1] + '/' + 'isca_parallel_test.sh', 'w+') as f:
                f.writelines(data)
               
    else:    
         
        if(environment == 'isca_test' or environment == 'isambard_test' or environment == 'isca_test_mesh' or environment == 'local'):
            n_dim = len(trays.ub)
            n_samples = 17500
        else:    
            n_dim = len(trays.ub)
            n_samples = 17500  # 50000: 63, 40000: 39, 48000: 46, 49000:67, 48500:48, 48750:56, 48740: 60, 48730: 53, 48731: 53 48735: 51
            # 11*5 = 55; 11*5-1 = 54
        #######################
        # write constraints.npz
        #######################
        if(environment == 'isca_test_mesh'):
            # decision and checkMesh is null
            initial_X = [] # decision vector
            initial_G = [] # geometric constraint
            initial_M = [] # checkMesh
            initial_C = [] # convergence
            
            file_constraints = 'constraints.npz'
            
            subprocess.call(['rm', '-r', current + '/' + file_constraints])
            
            try:
                np.savez(current + '/' + file_constraints, initial_X, initial_M, initial_C, initial_G)
                print('Data saved in file: ', file_constraints)
            except Exception as e:
                print(e)
                print('Data saving failed.')  
        else:    
            # decision and checkMesh is null
            initial_X = [] # decision vector
            initial_G = [] # geometric constraint
            initial_M = [] # checkMesh
            initial_C = [] # convergence
            
            file_constraints = 'constraints.npz'
            
            subprocess.call(['rm', '-r', current + '/' + file_constraints])
            
            try:
                np.savez(current + '/' + file_constraints, initial_X, initial_M, initial_C, initial_G)
                print('Data saved in file: ', file_constraints)
            except Exception as e:
                print(e)
                print('Data saving failed.')    
        
        
        ################################
        # obtain Latin Hypercube samples
        ################################
        print("initialisation(): number of Latin Hypercube samples", str(n_samples))
        samples, samples_unconstrained = lhs_initial_samples(n_dim, trays.ub, trays.lb, n_samples, cfunc=gap_constraint, cargs=(trays,), ckwargs={})
        print('initialisation(): number of samples that pass constraints, ', len(samples))
        print('initialisation(): number of samples unconstrained, ', len(samples_unconstrained))
        print('initialisation(): samples that pass constraints, ', samples)
        ###################################################################
        # correct all files using the directory names and create geometries
        ###################################################################
        
        X_new=[]
        for decision in samples_unconstrained:
            if decision not in samples:
                X_new.append(decision)
                
        X_new=np.array(X_new)
        M_new=np.zeros((len(X_new),1), dtype=int)
        C_new=np.zeros((len(X_new),1), dtype=int)
        G_new=-1*np.ones((len(X_new),1), dtype=int)
        
        
        try:     
            np.savez(current + '/' + file_constraints, X_new, M_new, C_new, G_new)
        except Exception as ex:
            print(ex)
            print('Data saving failed to constraints file.')
        
        
        # create filedirs
        filedirs = []
        
        # loop through the list to create directories:
        for s in samples:
            
            # create a working directory from the sample:
            dir_name = "_"
            for j in range(len(s)):
                dir_name += "{0:.8f}_".format(s[j])
            
            # replace any directories containing []
            dir_name = dir_name.replace("[","")
            dir_name = dir_name.replace("]","")
            
            # add the name to a list of directories
            filedirs.append(current + case_path + dir_name)
            
            # create the directory from the last in the list and 
            subprocess.call(['mkdir', filedirs[-1] + '/'])
            
            # copy all source files into the newly created directory
            subprocess.call(['cp', '-r', current + source_path, filedirs[-1]])
            
               
            # write the decision vector to a file        
            with open(filedirs[-1] + "/decision_vector.txt", "w+") as myfile:
                for i in range(0,len(s)):
                    myfile.write(str(s[i])+ '\n')
            print("initialisation(): written decision vector to a file")
        
            if(environment == 'isca_test_mesh' or environment == 'local'):
                with open(filedirs[-1] + '/create_mesh.py', 'r') as f:
                    data = f.readlines()
            
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                        
                #write the lines to the file        
                with open(filedirs[-1] + '/create_mesh.py', 'w+') as f:
                    f.writelines(data)     
                    
            else:    
                
                with open(filedirs[-1] + '/run_case.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
            
                #write the lines to the file        
                with open(filedirs[-1] + '/run_case.py', 'w+') as f:
                    f.writelines(data)

            if(environment == 'isambard_test'):
                
                print("not creating geometry")
                
            else:

                with open(filedirs[-1] + '/create_pot_underflow_and_base.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_pot_underflow_and_base.py', 'w+') as f:
                    f.writelines(data) 
                
                with open(filedirs[-1] + '/create_tray_lip_1.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_tray_lip_1.py', 'w+') as f:
                    f.writelines(data)                  
       
                with open(filedirs[-1] + '/create_top_and_bottom.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_top_and_bottom.py', 'w+') as f:
                    f.writelines(data) 
                
                with open(filedirs[-1] + '/create_benching.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_benching.py', 'w+') as f:
                    f.writelines(data)                 
       
                with open(filedirs[-1] + '/create_benching_circle.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_benching_circle.py', 'w+') as f:
                    f.writelines(data)                
                   
                with open(filedirs[-1] + '/create_vessel.py', 'r') as f:
                    data = f.readlines()
                
                # change the line to the correct directory
                for line in range(len(data)):       
                    if 'dir_name = None' in data[line]:
                        data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
                
                #write the lines to the file        
                with open(filedirs[-1] + '/create_vessel.py', 'w+') as f:
                    f.writelines(data) 
                
                #case_path = '/case/'
                
                a_top = 440.0
                b_top = 440.0
                x_top = 0.0
                z_top = 0.0
                    
                x_bottom = 0.0
                y_bottom = s[0]
                angle_bottom = 0.0
                a_bottom = 50.0
                b_bottom = 50.0
                alpha = s[1]
                beta = s[2]
                
                x = s
            
                ################################################################################
                ## TRAY 0
                ################################################################################
                tray = 0              
                file_name_tray = "tray_" + str(tray) + ".stl"
                stl_benching(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name_tray, x)
                   
                print("tray_0.stl created")
                     
                ################################################################################
                ## TRAY 1
                ################################################################################
                tray = 1
                file_name_tray = "tray_" + str(tray) + ".stl"
                stl_tray(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name_tray, x)    
                
                print("tray_1.stl created")
                
                ################################################################################
                ## LIP 1
                ################################################################################    
                file_name_lip = "lip_" + str(tray) + ".stl"
                stl_lip(x_bottom, y_bottom, angle_bottom, a_bottom, b_bottom, file_name_lip, x)
                
                print("lip_1.stl created")  
                
                if(environment == 'local'):
                
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_pot_underflow_and_base.py'])
                    
                    print("salome vessel.stl created")   
                    
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_pot_underflow_and_base.py'])
                    
                    print("salome vessel.stl created")    
                 
                if(environment == 'local'):
                
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_tray_lip_1.py'])
                    
                    print("salome tray_lip_1.stl created")
                
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_tray_lip_1.py'])
                
                    print("salome tray_lip_1.stl created")
                
                if(environment == 'local'):
                
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_top_and_bottom.py'])
                    
                    print("salome top and bottom stls created") 
                
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_top_and_bottom.py'])
                   
                    print("salome top and bottom stls created")    
                
                if(environment == 'local'):
            
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_benching.py'])
                    
                    print("salome benching created") 
                
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_benching.py'])
                   
                    print("salome benching created") 
                
                if(environment == 'local'):
            
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_benching_circle.py'])
                    
                    print("salome benching circle created") 
                
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_benching_circle.py'])
                   
                    print("salome benching circle created")     
                    
                if(environment == 'local'):
            
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/create_vessel.py'])
                    
                    print("salome vessel created") 
                
                elif((environment == 'isca') or (environment == 'isca_test')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/create_vessel.py'])
                   
                    print("salome vessel created")                    

                if(environment == 'isambard' or environment == 'isambard_test'):
                    
                    if (process == 'BO'):                    
                    
                        # change bash_pointwise in order to run from current directory
                        with open(current + case_path + dir_name + '/salome_directory.sh', 'r') as myfile:
                            data_myfile = myfile.readlines()
                
                        # read in each line and change the directory location                
                        for line in range(len(data_myfile)):       
                            if 'export SALOME_DIRECTORY=' in data_myfile[line]:
                                data_myfile[line] = 'export SALOME_DIRECTORY=' + '"'+ current + case_path + dir_name + '"' + '\n'       
                        
                        with open(current + case_path + dir_name + '/salome_directory.sh', 'w+') as myfile:
                            myfile.writelines(data_myfile)
                    
                        # copy pointwise_directory.sh to home
                        shutil.copyfile(current + case_path + dir_name + '/salome_directory.sh', home + '/salome_directory.sh')  
                        
                        # make file executable
                        subprocess.call(['chmod', '755', home + '/salome_directory.sh'])
                                
                        print("written salome_directory.sh to a file")
                                
                                 
                        # name of the script to submit pointwise with
                        bash_salome = 'salome_script.sh'
                        
                        # change bash_pointwise in order to run from current directory
                        with open(current + case_path + dir_name + '/' + bash_salome, 'r') as f:
                            data = f.readlines()
                        
                        # read in each line and change the directory location                
                        for line in range(len(data)):       
                            if '#PBS -o' in data[line]:
                                data[line] = '#PBS -o 10.141.0.1:' + current + case_path + dir_name + '/log.salome' + '\n'
                        
                            if '#PBS -e' in data[line]:                                                                                                                                                 
                                data[line] = '#PBS -e 10.141.0.1:' + current + case_path + dir_name + '/err.salome' + '\n'
                                
                        # write the changes to bash_pointwise        
                        with open(current + case_path + dir_name + '/' + bash_salome, 'w+') as f:
                            f.writelines(data)
                            
                        start_sim = time.time()                    
                        # use ssh to send the job to phase 1
                        subprocess.call(['ssh','ex-aroberts@login-01.gw4.metoffice.gov.uk','source $HOME/salome_directory.sh; export PBS_HOME=/cm/shared/apps/pbspro/var/spool/pbspro; export PBS_EXEC=/cm/shared/apps/pbspro/19.2.8.20200925072630; export PBS_SERVER=gw4head; /cm/shared/apps/pbspro/19.2.8.20200925072630/bin/qsub $SALOME_DIRECTORY/salome_script.sh;'], cwd=current + case_path + dir_name,\
                        stdout = open(current + case_path + dir_name + '/log.ssh', 'w+'), \
                        stderr = open(current + case_path + dir_name + '/err.ssh', 'w+'))
            
                        # check if the log files exist, if not sleep for 1 minute
                        while not (os.path.exists(current + case_path + dir_name + '/' + 'log.salome') and os.path.exists(current + case_path + dir_name + '/' + 'err.salome')):
                            time.sleep(60)
                            
                        print("salome completed in ", (time.time()-start_sim)/60.0, " min")
            
                    else:
                        
                        # change bash_pointwise in order to run from current directory
                        with open(current + case_path + dir_name + '/salome_directory_lhs.sh', 'r') as myfile:
                            data_myfile = myfile.readlines()
                
                        # read in each line and change the directory location                
                        for line in range(len(data_myfile)):       
                            if 'export SALOME_DIRECTORY_LHS=' in data_myfile[line]:
                                data_myfile[line] = 'export SALOME_DIRECTORY_LHS=' + '"'+ current + case_path + dir_name + '"' + '\n'       
                        
                        with open(current + case_path + dir_name + '/salome_directory_lhs.sh', 'w+') as myfile:
                            myfile.writelines(data_myfile)
                        
                        # copy pointwise_directory.sh to home
                        shutil.copyfile(current + case_path + dir_name + '/salome_directory_lhs.sh', home + '/salome_directory_lhs.sh')  
                        
                        # make file executable
                        subprocess.call(['chmod', '755', home + '/salome_directory_lhs.sh'])
                                
                        print("written salome_directory_lhs.sh to a file")
                                
                                 
                        # name of the script to submit pointwise with
                        bash_salome = 'salome_script_lhs.sh'
                        
                        # change bash_pointwise in order to run from current directory
                        with open(current + case_path + dir_name + '/' + bash_salome, 'r') as f:
                            data = f.readlines()
                        
                        # read in each line and change the directory location                
                        for line in range(len(data)):       
                            if '#PBS -o' in data[line]:
                                data[line] = '#PBS -o 10.141.0.1:' + current + case_path + dir_name + '/log.salome' + '\n'
                        
                            if '#PBS -e' in data[line]:                                                                                                                                                 
                                data[line] = '#PBS -e 10.141.0.1:' + current + case_path + dir_name + '/err.salome' + '\n'
                                
                        # write the changes to bash_pointwise        
                        with open(current + case_path + dir_name + '/' + bash_salome, 'w+') as f:
                            f.writelines(data)
                            
                        start_sim = time.time()                    
                        # use ssh to send the job to phase 1
                        subprocess.call(['ssh','ex-aroberts@login-01.gw4.metoffice.gov.uk','source $HOME/salome_directory_lhs.sh; export PBS_HOME=/cm/shared/apps/pbspro/var/spool/pbspro; export PBS_EXEC=/cm/shared/apps/pbspro/19.2.8.20200925072630; export PBS_SERVER=gw4head; /cm/shared/apps/pbspro/19.2.8.20200925072630/bin/qsub $SALOME_DIRECTORY_LHS/salome_script_lhs.sh;'], cwd=current + case_path + dir_name,\
                        stdout = open(current + case_path + dir_name + '/log.ssh', 'w+'), \
                        stderr = open(current + case_path + dir_name + '/err.ssh', 'w+'))                        
            
            
                        # check if the log files exist, if not sleep for 1 minute
                        while not (os.path.exists(current + case_path + dir_name + '/' + 'log.salome') and os.path.exists(current + case_path + dir_name + '/' + 'err.salome')):
                            time.sleep(60)
                            
                        print("salome completed in ", (time.time()-start_sim)/60.0, " min")
                
                # Kill any Salome processes at the end
                
                if(environment == 'local'):
                    
                    subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', 'killall'])   
                
                elif((environment == 'isca' or environment == 'isca_test') or (environment == 'isca_test_mesh')):
                
                    subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', 'killall'])
        
    ###########################
    # write initial_samples.npz
    ###########################
    
    # read decision vector and write to npz file
    print('Writing empty initial_samples.npz file...')
    
    # hyper volume improvement is null
    hpv = []
    initial_time = 0
    initial_X = []
    initial_Y = []
    #initial_convergence = []
    
    # the name of the sim_file is initial_samples.npz    
    sim_file = 'initial_samples.npz'
    
    # remove npz file if it exists
    subprocess.call(['rm', '-r', current + '/' + sim_file]) 
    
    # initial_X is decision vector
    # initial_Y is 2 objectives
    # hpv is hypervolume improvement
    # initial_time is zero
    # initial_convergence is convergence failure  
    try:
        np.savez(current + '/' + sim_file, initial_X, initial_Y, hpv, initial_time)
        print('Data saved in file: ', sim_file)
    except Exception as e:
        print(e)
        print('Data saving failed.')

    #############################
    # edit the submission scripts
    #############################
    
    # decide environment and adjust bash script for running.
    if (((environment == 'isca') or (environment == 'isca_test')) or (environment == 'isca_test_mesh')):
        # if the environment is isca
        for d in range(len(filedirs)):
                                    
            # open the bash script for running
            with open(filedirs[d] + '/' + bash, 'r') as f:
                data = f.readlines()
                
            # change the line to the correct directory
            for line in range(len(data)):       
                if '#SBATCH -D' in data[line]:
                    data[line] = '#SBATCH -D '+ filedirs[d]+'/' + '\n'
            
            #write the lines to the file        
            with open(filedirs[d] + '/' + bash, 'w+') as f:
                f.writelines(data)
                
    elif ((environment == 'isambard') or (environment == 'isambard_test')):
        # if the environment is isambard
        for d in range(len(filedirs)):
            
            # open the bash script for running
            with open(filedirs[d] + '/' + bash, 'r') as f:
                data = f.readlines()
                
            # change the line to the correct directory
            for line in range(len(data)):       
                if '#PBS -d' in data[line]:
                    data[line] = '#PBS -d '+ filedirs[d]+'/' + '\n'
            
            #write the lines to the file        
            with open(filedirs[d] + '/' + bash, 'w+') as f:
                f.writelines(data)
                
    else:
        # if the environment is local:    
        
        print("locally do nothing")
            
    print('Total number of simulations, ', len(filedirs))
    print('All directories created.')

    return filedirs


#############################################################
# lhs_initial_samples() FUNCTION USED BY INITAL_SAMPLING ONLY
#############################################################

def lhs_initial_samples(n_dim, ub, lb, n_samples=4, cfunc=None, cargs=(), ckwargs={}):
    """
    Generate Latin hypercube samples from the decision space using pyDOE.

    Parameters.
    -----------
    n_samples (int): the number of samples to take. 
    cfunc (method): a cheap constraint function.
    cargs (tuple): arguments for cheap constraint function.
    ckawargs (dictionary): keyword arguments for cheap constraint function. 

    Returns a set of decision vectors.         
    """
    seed = 1270 # was 1435
    np.random.seed(seed)
    samples = LHS(n_dim, samples=n_samples)
    
    scaled_samples_unconstrained = ((ub - lb) * samples) + lb            
        
    if cfunc is not None: # check for constraints
        print('Checking for constraints.')
        scaled_samples = np.array([i for i in scaled_samples_unconstrained if cfunc(i, *cargs, **ckwargs)])

    return scaled_samples, scaled_samples_unconstrained



###############################################################################
## BAYESIAN OPTIMISATION FUNCTIONS
###############################################################################

##############################################
# BO_submit(x) FUNCTION USED BY OPTIMISER ONLY
##############################################

def BO_submit(x, ductPolygon, bafflePolygon, cornerPolygon):
    """
    Submit the job to the relevant queue without a python queue 
    Copy of no_queue without a list of directories - i.e. it just submits one run.

    Input
    -----
    A single directory. 
    
    Output
    ------
    Does not return anything.         
    """
    # get current directory
    current = os.getcwd()
    
    # set the location of the case path
    case_path = "/data/HeadCell/case/"
    
    dir_name = "_"
    for j in range(len(x)):
        dir_name += "{0:.3f}_".format(x[j])
    
    # replace any directories containing []
    dir_name = dir_name.replace("[","")
    dir_name = dir_name.replace("]","")
    
    BO_create_geometry(x, ductPolygon, bafflePolygon, cornerPolygon)
    
    # submit job using msub if on isca:
    if ((environment == 'isca') or (environment == 'isca_test')):
        start_msub = time.time()
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        subprocess.call(['python','-u', current + case_path + dir_name + '/' + 'run_case.py'], cwd=current + case_path + dir_name, \
              stdout = open(current + case_path + dir_name + '/log.subprocess', 'w+'), \
              stderr = open(current + case_path + dir_name + '/err.subprocess', 'w+'))
        
#        while (os.stat(current + case_path + dir_name + "/err.subprocess").st_size != 0):
#            print("did not successfully submit: deleting all jobs, log.subprocess and err.subprocess and resubmitting")
#            time.sleep(100)
#            subprocess.call(['scancel', '-u', 'apr207'])
#            subprocess.call(['rm', current + case_path + dir_name + '/log.subprocess'])
#            subprocess.call(['rm', current + case_path + dir_name + '/err.subprocess'])
#            subprocess.call(['sbatch', current + case_path + dir_name + '/' + bash], cwd=current + case_path + dir_name, \
#              stdout = open(current + case_path + dir_name + '/log.subprocess', 'w+'), \
#              stderr = open(current + case_path + dir_name + '/err.subprocess', 'w+'))
        
#            if (os.stat(current + case_path + dir_name + "/err.subprocess").st_size != 0):
#                print("did not successfully submit: deleting all jobs, log.subprocess and err.subprocess and resubmitting")
#                time.sleep(100)
#                subprocess.call(['scancel', '-u', 'apr207'])
#                subprocess.call(['rm', current + case_path + dir_name + '/log.subprocess'])
#                subprocess.call(['rm', current + case_path + dir_name + '/err.subprocess'])
#                subprocess.call(['sbatch', current + case_path + dir_name + '/' + bash], cwd=current + case_path + dir_name, \
#                  stdout = open(current + case_path + dir_name + '/log.subprocess', 'w+'), \
#                  stderr = open(current + case_path + dir_name + '/err.subprocess', 'w+'))
            
        while not os.path.exists(current + case_path + dir_name + '/' + 'completed.txt'):
            time.sleep(60)
        now = datetime.now()
        end_time = now.strftime("%H:%M:%S")
        print("Start time, " + start_time + " End time, " + end_time + " Elapsed time, " + str(time.time() - start_msub))
        
    # submit job using qsub if on isambard - must IMMEDIATELY return, unlike mul function which uses a queue:
    elif ((environment == 'isambard') or (environment == 'isambard_test')):
        start_msub = time.time()
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        subprocess.call(['qsub', current + case_path + dir_name + '/' + bash], cwd=current + case_path + dir_name, \
              stdout = open(current + case_path + dir_name + '/log.subprocess', 'w+'), \
              stderr = open(current + case_path + dir_name + '/err.subprocess', 'w+'))

        while not os.path.exists(current + case_path + dir_name + '/' + 'completed.txt'):
            time.sleep(60)
        now = datetime.now()
        end_time = now.strftime("%H:%M:%S")
        print("Start time, " + start_time + " End time, " + end_time + " Elapsed time, " + str(time.time() - start_msub))

    # submit job using shell locally:    
    else:
        start_msub = time.time()
        now = datetime.now()
        start_time = now.strftime("%H:%M:%S")
        subprocess.call(['sh', current + case_path + dir_name + '/' + bash], cwd=current + case_path + dir_name, \
              stdout = open(current + case_path + dir_name + '/log.subprocess', 'w+'), \
              stderr = open(current + case_path + dir_name + '/err.subprocess', 'w+'))
        now = datetime.now()
        end_time = now.strftime("%H:%M:%S")
        
        
    print("Start time, " + start_time + " End time, " + end_time + " Elapsed time, " + str(time.time() - start_msub))     
        
    case_path = current + case_path + dir_name   
  
    data_cp = np.genfromtxt(case_path + '/cp.txt', delimiter=' ', skip_header=0, names=['cp'])
    cp = data_cp['cp']
    data_cf = np.genfromtxt(case_path + '/cf.txt', delimiter=' ', skip_header=0, names=['cf'])
    cf = data_cf['cf']     
    
    case_path = "/data/HeadCell/case/" 
    completed_path = "/data/HeadCell/completed/" 
    seeding()
                                
    random_location = randint(0,200000000)
    # remove directory
    subprocess.call(['cp', '-rf', current + case_path + dir_name, current + completed_path + dir_name + str(random_location)])
    subprocess.call(['rm', '-rf', current + case_path + dir_name])
    
    return cp, -1.0*cf

######################################################
# BO_create_geometry() FUNCTION USED BY OPTIMISER ONLY
######################################################
            
def BO_create_geometry(x, ductPolygon, bafflePolygon, cornerPolygon):
    """
    Samples the design space,
    Writes the decision vector to a file
    Creates an empty initial samples file
    Copies and edits the qsub or msub submission script.

    Parameters.
    -----------
    There are no parameters. 

    Returns a list of directories for submission.         
    """
    start_time = time.time()
    
    # get current directory
    current = os.getcwd()
    


    ######################
    # CREATE DIRECTORIES #
    ######################

    # create filedirs
    filedirs = []

    #stl_dir = "/constant/triSurface/"
    
    # loop through the list to create directories:
            
    s = x
    
    # create a working directory from the sample:
    dir_name = "_"
    for j in range(len(s)):
        dir_name += "{0:.3f}_".format(s[j])
    
    # replace any directories containing []
    dir_name = dir_name.replace("[","")
    dir_name = dir_name.replace("]","")
    
    # add the name to a list of directories
    filedirs.append(current + case_path + dir_name)
    
    # create the directory from the last in the list and 
    subprocess.call(['mkdir', filedirs[-1] + '/'])
    
    # copy all source files into the newly created directory
    subprocess.call(['cp', '-r', current + source_path + '.', filedirs[-1]])

    with open(filedirs[-1] + "/decision_vector.txt", "w+") as myfile:
        for i in range(0,len(s)):
            myfile.write(str(s[i])+ '\n')
                    
                
                
    ##############
    # CREATE STL #
    ##############
    ctrl_pts_duct = ductPolygon.convert_dec2pos(s[:n_dim])   # the four ends points and the single decision
    duct_curve_points = ductPolygon.subdc_to_points([ctrl_pts_duct], niterations)
    print("duct_curve_points", len(duct_curve_points))
    
    ctrl_pts_baffle = bafflePolygon.convert_dec2pos(s[n_dim:])
    baffle_curve_points = bafflePolygon.subdc_to_points([ctrl_pts_baffle], niterations)
    print("baffle_curve_points", len(baffle_curve_points))

    ctrl_pts_corner = cornerPolygon.convert_dec2pos_corner(s[n_dim:])
    corner_curve_points = cornerPolygon.subdc_to_points([ctrl_pts_corner], niterations)
    print("corner_curve_points", len(corner_curve_points))


    # write the decision vector to a file        
    with open(filedirs[-1] + "/duct_curve_points.txt", "w+") as myfile1:
        for i in range(0,len(duct_curve_points)):
            myfile1.write(str(duct_curve_points[i][0]) + ',' + str(duct_curve_points[i][1]) + '\n')
    print("written duct curve points to duct_curve_points.txt")


    with open(filedirs[-1] + "/baffle_curve_points.txt", "w+") as myfile2:
        for i in range(0,len(baffle_curve_points)):
            myfile2.write(str(baffle_curve_points[i][0]) + ',' + str(baffle_curve_points[i][1]) + '\n')
    print("written baffle curve points to baffle_curve_points.txt")

    with open(filedirs[-1] + "/corner_curve_points.txt", "w+") as myfile3:
        for i in range(0,len(corner_curve_points)):
            myfile3.write(str(corner_curve_points[i][0]) + ',' + str(corner_curve_points[i][1]) + '\n')
    print("written corner curve points to corner_curve_points.txt")
    
    
    with open(filedirs[-1] + '/headcell_geometry.py', 'r') as f:
        data = f.readlines()
    
    # change the line to the correct directory
    for line in range(len(data)):       
        if 'dir_name = None' in data[line]:
            data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
    
    #write the lines to the file        
    with open(filedirs[-1] + '/headcell_geometry.py', 'w+') as f:
        f.writelines(data)
    
    
    if(environment == 'local'):

        subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/headcell_geometry.py'])
        
        print("salome walls.stl created")   
        
        subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', 'killall'])   
        
    elif((environment == 'isca') or (environment == 'isca_test_mesh')):

        subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/headcell_geometry.py'])
        
        print("salome walls.stl created") 
        
        subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', 'killall'])
    
    




























#    # create filedirs
#    filedirs = []
#    
#    thickness = np.array([0, 0, 103.0])
#    stl_file_name_duct = 'lowerDuct.stl'
#    stl_file_name_baffle_1 = 'corner_1.stl'
#    stl_file_name_baffle_2 = 'corner_2.stl'
#    stl_file_name_baffle_3 = 'corner_3.stl'
#    stl_file_name_baffle_4 = 'corner_4.stl'
#    stl_file_name_baffle_5 = 'corner_5.stl'
#    stl_file_name_baffle_6 = 'corner_6.stl'
#    
#    offset_2_y = np.array([0.0, 87.0])
#    offset_2345_x = np.array([0.0, 0.0])
#    offset_3_y = offset_2_y + offset_2_y
#    offset_4_y = offset_2_y + offset_2_y + offset_2_y
#    offset_5_y = offset_2_y + offset_2_y + offset_2_y + offset_2_y
#    offset_6_y = offset_2_y + offset_2_y + offset_2_y + offset_2_y + offset_2_y
#    offset_6_x = np.array([-25.0, 0.0])
#    
#    stl_dir = "/constant/triSurface/"
#    
#    # loop through the list to create directories:
#    #for s in lhs_samples:
#        
#    # create a working directory from the sample:
#    dir_name = "_"
#    for j in range(len(x)):
#        dir_name += "{0:.3f}_".format(x[j])
#    
#    # replace any directories containing []
#    dir_name = dir_name.replace("[","")
#    dir_name = dir_name.replace("]","")
#    
#    case_path = '/data/HeadCell/case/'
#    
#    # add the name to a list of directories
#    filedirs.append(current + case_path + dir_name)
#    
#    # create the directory from the last in the list and 
#    subprocess.call(['mkdir', filedirs[-1] + '/'])
#    
#    # copy all source files into the newly created directory
#    subprocess.call(['cp', '-r', current + source_path + '.', filedirs[-1]])
#
#    with open(filedirs[-1] + "/decision_vector.txt", "w+") as myfile:
#        for i in range(0,len(x)):
#            myfile.write(str(x[i])+ '\n')
#                
#    ##############
#    # CREATE STL #
#    ##############
#    ctrl_pts_duct = ductPolygon.convert_dec2pos(x[:2])
#    
#    ductPolygon.subdc_to_stl_mult([ctrl_pts_duct], niterations,\
#                        thickness = thickness,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_duct,\
#                        draw = False)
#    
#    ctrl_pts_baffle_1 = bafflePolygon.convert_dec2pos(x[2:]) 
#    
#    bafflePolygon.subdc_to_stl_mult([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_1,\
#                        draw = False)    
#     
#    bafflePolygon.subdc_to_stl_mult_offset([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        offset_x = offset_2345_x,\
#                        offset_y = offset_2_y,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_2,\
#                        draw = False)  
#
#    bafflePolygon.subdc_to_stl_mult_offset([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        offset_x = offset_2345_x,\
#                        offset_y = offset_3_y,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_3,\
#                        draw = False)     
#
#    bafflePolygon.subdc_to_stl_mult_offset([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        offset_x = offset_2345_x,\
#                        offset_y = offset_4_y,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_4,\
#                        draw = False)
#
#    bafflePolygon.subdc_to_stl_mult_offset([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        offset_x = offset_2345_x,\
#                        offset_y = offset_5_y,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_5,\
#                        draw = False)
#
#    bafflePolygon.subdc_to_stl_mult_offset([ctrl_pts_baffle_1], niterations,\
#                        thickness = thickness,\
#                        offset_x = offset_6_x,\
#                        offset_y = offset_6_y,\
#                        file_directory = filedirs[-1]+'/case_source'+stl_dir, file_name = stl_file_name_baffle_6,\
#                        draw = False)
#            
    print('Prepared geometry for 1 simulation total elapsed time, ' + str(time.time() - start_time))


##############################################################
# gap_constraint(x, layout) FUNCTION USED BY OPTIMISER AND LHS
##############################################################

def duct_and_baffle_constraint(x, ductPolygon, bafflePolygon, corner_boundary, baffle_boundary, duct_boundary):
    """
    Checks only the gap constraint: writes no constraints.npz file.

    Parameters.
    -----------
    x (numpy array): the decision vector. 
    layout (Ellipse object): the object for the generation of the pointwise files. 

    Returns whether the constraint was successfully passed.         
    """

    # THIS IS A COPY OF THE GAP CONSTRAINT FROM gap_and_checkMesh_constraint

    # defaults for success:
    duct_success = ductPolygon.duct_constraint_check(x[:2], corner_boundary, baffle_boundary)

    baffle_success = bafflePolygon.baffle_constraint_check(x[2:], duct_boundary)

    return (duct_success and baffle_success)

#def only_duct_constraint(x, ductPolygon, corner_boundary, baffle_boundary):
#    """
#    Checks only the gap constraint: writes no constraints.npz file.
#
#    Parameters.
#    -----------
#    x (numpy array): the decision vector. 
#    layout (Ellipse object): the object for the generation of the pointwise files. 
#
#    Returns whether the constraint was successfully passed.         
#    """
#
#    # THIS IS A COPY OF THE GAP CONSTRAINT FROM gap_and_checkMesh_constraint
#
#    # defaults for success:
#    duct_success = ductPolygon.duct_constraint_check(x[:6], corner_boundary, baffle_boundary)
#
#    #baffle_success = bafflePolygon.baffle_constraint_check(x, duct_boundary)
#
#    return duct_success
#
#def only_baffle_constraint(x, bafflePolygon, duct_boundary):
#    """
#    Checks only the gap constraint: writes no constraints.npz file.
#
#    Parameters.
#    -----------
#    x (numpy array): the decision vector. 
#    layout (Ellipse object): the object for the generation of the pointwise files. 
#
#    Returns whether the constraint was successfully passed.         
#    """
#
#    # THIS IS A COPY OF THE GAP CONSTRAINT FROM gap_and_checkMesh_constraint
#
#    # defaults for success:
#    #duct_success = ductPolygon.duct_constraint_check(x, corner_boundary, baffle_boundary)
#
#    baffle_success = bafflePolygon.baffle_constraint_check(x[6:], duct_boundary)
#
#    return baffle_success


import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits import mplot3d

class ControlPolygon2D(object):
    """
    The ControlPolygon2D class defines the control bounadary within which
    any valid curve must reside. The vertices of the control region is accepted
    as inputs, where the sequence specifies the polygon; as such the first and
    the last points must be the same to construct a closed polygon.

    For instance, a polygon with points a, b, c and d may be represented as:
    [a, b, c, d, a]; this must construct a non-intersecting polygon. Note that,
    each point should contain horizontal and vertical coordinates,
    e.g. a = [ax, ay].

    The end points for the subdivision curve should be included in this control
    polygon. The indices of these end points must be specified. For instance,
    if a dn c are the end points for the subdivion curve, then they should be
    specified by setting fixed_points = [[0], [2]]. In the interest of
    smoothness multiple end points may be grouped together (and specified
    through their indicies) so that the curve remains differentiable where it
    leaves the surface of the base shape.

    """

    def __init__(self, vertices, fixed_points, num_points, corner_constraint, baffle_constraint, duct_constraint, niterations, \
                 boundary=None, dim=2):
        '''
        Args:
            vertices (numpy array): vertices of the control region.
            fixed_points (list - 2D): indices of the end points for the
                                        subdivion curves.
            num_points (int): number of control points for any subdivision
                                curve within this polygon.

        Kwargs:
            boundary (numpy array): if a specific rectangular region within
                                    the control polygon is sought wihtin which
                                    any curve will be generated. This should be
                                    structured as follows: [min_x, min_y,
                                    max_x, max_y].
            dim (int): number of dimensions. It should always be 2 in this
                        particular case.

        '''
        self.polygon = Path(vertices)
        self.vertices = self.polygon.vertices
        self.fixed = fixed_points
        self.num_points = num_points
        self.boundary = boundary
        self.dim = dim
        self.baffle_constraint = baffle_constraint
        self.corner_constraint = corner_constraint
        self.duct_constraint = duct_constraint
        self.niterations = niterations

    def vector_projection_rearrange(self, points):
        ab = self.vertices[self.fixed[1][0]] - self.vertices[self.fixed[0][-1]]
        ap = points - self.vertices[self.fixed[0][-1]]
        proj = np.dot(ap, ab.T)/np.linalg.norm(ab)
        #print (proj, np.argsort(proj))
        return np.vstack([self.vertices[self.fixed[0]], \
                        points[np.argsort(proj)], \
                        self.vertices[self.fixed[1]]])


    #######################
    # edited for 125 micron
    #######################
    def create_duct_boundary_constraint(self):
        
        # lower part of baffle
#        start = -16.942134+2.158 # giving 1.25 times baffle height of length
#        stop  = -17.250
#        x_duct_bottom = np.linspace(start, stop, num=20, endpoint=True)[:,None]
#        
#        y_duct_bottom = (19.668)*np.ones_like(x_duct_bottom)
#        data_points = np.concatenate((x_duct_bottom, y_duct_bottom), axis=1)
        
        # linear part of baffle
        startx = -17.250 
        stopx = -34.851
        
        x_duct_linear = np.linspace(startx, stopx, num=100, endpoint=True)[:,None]
        
        starty = 19.668
        stopy = 47.770
        
        y_duct_linear = np.linspace(starty, stopy, num=100, endpoint=True)[:,None]
        
#        x_values = np.concatenate((x_duct_bottom, x_duct_linear))
#        y_values = np.concatenate((y_duct_bottom, y_duct_linear))
#
#        data_points = np.concatenate((x_values, y_values), axis=1)


        data_points = np.concatenate((x_duct_linear, y_duct_linear), axis=1)



        return data_points    

    #######################
    # edited for 125 micron
    #######################
    def create_corner_boundary_constraint(self):
        
        # lower part of baffle
        start = -16.942134+2.158 # giving 1.25 times baffle height of length
        stop  = -16.239
        x_baffle_bottom = np.linspace(start, stop, num=20, endpoint=False)[:,None]
        
        # linear part of baffle
        start = -16.239          # done
        stop  = -16.942134       # done
        x_baffle_linear = np.linspace(start, stop, num=10, endpoint=True)[:,None]
        x_values = np.concatenate((x_baffle_bottom, x_baffle_linear))
    
        # lower part of baffle
        y_baffle_bottom = (59.197)*np.ones_like(x_baffle_bottom) # done

        # linear part of baffle
        y_baffle_linear = (( (59.840578 - 59.197 ))/((-17.25 - - 16.239)))*(x_baffle_linear - -17.25) + (59.840578) # 883.7 offset because of inlet
    
        y_values = np.concatenate((y_baffle_bottom, y_baffle_linear))
        
        # collect x and y
        data_points = np.concatenate((x_values, y_values), axis=1)
    
        return data_points

    #######################
    # edited for 125 micron
    #######################
    def create_baffle_boundary_constraint(self):
        
        # lower part of baffle
        start = -16.942134+2.158 # giving 1.25 times baffle height of length
        stop = -16.124418 # start of radius
        
        x_baffle_bottom = np.linspace(start, stop, num=20, endpoint=False)[:,None]
        
        #small radius
        x_radius = np.array([-16.124418,    -16.176188,   -16.224667,  -16.266774,   -16.299834])[:,None]
        
        # linear part of baffle
        startx = -16.299834
        stopx = -16.942134
        x_baffle_linear = np.linspace(startx, stopx, num=20, endpoint=True)[:,None]

        #large radius
        x_radius_large = np.array([-16.942134,         -16.985526, -17.002259, -16.991153, -16.952992, -16.890465, -16.807982, -16.711357,       -16.607403])[:,None]

        start = -16.607403
        stop = -16.942134+2.158
        x_baffle_top = np.linspace(start, stop, num=20, endpoint=True)[:,None]

        # concatenate bottom and radius and top
        x_values = np.concatenate((x_baffle_bottom, x_radius, x_baffle_linear, x_radius_large, x_baffle_top))
        
        # lower part of baffle
        y_baffle_bottom = (21.379)*np.ones_like(x_baffle_bottom) # 883.7 offset because of inlet
        
        y_radius = np.array([21.379, 21.385578, 21.404894, 21.435721, 21.476099])[:,None]
        
        # linear part of baffle
        start = 21.476099
        stop = 22.501285     
        
        # linear part of baffle
        y_baffle_linear = (( (stop - start))/((stopx - startx)))*(x_baffle_linear - startx) + start # 883.7 offset because of inlet
    
        #large radius
        y_radius_large = np.array([22.501285,       22.59677, 22.700309,     22.804601,     22.902295, 22.986501,    23.051284,    23.092076,       23.106])[:,None]
    
        y_baffle_top = (23.106)*np.ones_like(x_baffle_top)
    
        # concatenate bottom and radius and top        
        y_values = np.concatenate((y_baffle_bottom, y_radius, y_baffle_linear, y_radius_large, y_baffle_top )) # 883.7 offset because of inlet
        
        # collect x and y
        data_points_1 = np.concatenate((x_values, y_values), axis=1)
        
        # offset data points
        y_values = y_values + 3.438
        
        data_points_2 = np.concatenate((x_values, y_values), axis=1)
        
        y_values = y_values + 3.438
        
        data_points_3 = np.concatenate((x_values, y_values), axis=1)
        
        y_values = y_values + 3.438
        
        data_points_4 = np.concatenate((x_values, y_values), axis=1)

        y_values = y_values + 3.438
        
        data_points_5 = np.concatenate((x_values, y_values), axis=1)

        y_values = y_values + 3.438
        
        data_points_6 = np.concatenate((x_values, y_values), axis=1)
        
        y_values = y_values + 3.438
        
        data_points_7 = np.concatenate((x_values, y_values), axis=1)
        
        y_values = y_values + 3.438
        
        data_points_8 = np.concatenate((x_values, y_values), axis=1)

        y_values = y_values + 3.438
        
        data_points_9 = np.concatenate((x_values, y_values), axis=1)

        y_values = y_values + 3.438
        
        data_points_10 = np.concatenate((x_values, y_values), axis=1)

        y_values = y_values + 3.438
        
        data_points_11 = np.concatenate((x_values, y_values), axis=1)

        data_points = np.concatenate((data_points_1, data_points_2, data_points_3, data_points_4, data_points_5, data_points_6, data_points_7, data_points_8, data_points_9, data_points_10, data_points_11), axis=0)
    
        return data_points
    
    

    
    def convert_dec2pos(self, x):
        mod_x = np.reshape(x, (-1, 2))
        return self.vector_projection_rearrange(mod_x)


    def convert_dec2pos_corner(self, x):
        mod_x = np.reshape(x, (-1, 2))
        
        points = self.vector_projection_rearrange(mod_x)
        
        for i in range(len(points)):
            if(i==2):
                points[i]  = points[i] + np.array([0,11*3.438])
                points[i]  = points[i]  - np.array([0,59.197])
                points[i]  = points[i]  / np.array([1,corner_scale_factor])
                points[i]  = points[i]  + np.array([0,59.197])
    
        return points
    
    


    def convert_pos2dec(self, pos):
        return np.concatenate(pos[len(self.fixed[0]):self.num_points+len(self.fixed[0])])

    def subdc_to_stl_mult(self, individuals, n_iter, thickness=np.array([0,0,1]),
                            file_directory=None, file_name=None, draw=False):
        """
        A method to save multiple subdivision curves in a single STL file. This may be 
        used for multiple control regions.
        
        Args:
            individuals (numpy array): DEAP control point individuals.
            n_iter (int): Number of subdivision iterations. Higher number will 
                            produce a smoother curve.
        
        Kwargs:
            thickness (numpy array): The thickness along z-axis (last element in the 
                                        array). It can also be used to offset 
                                        x- and y-axis.
            file_directory (str): The destination directory for the STL file. 
            file_name (str): The name of the STL file. 
            draw (bool): Control for drawing graphs.
        """
        all_data = [] 
        # generate faces for each individual curve in the DEAP inidividuals
        for control_points in individuals:
            data_points = self.catmull(control_points)
            for i in range(n_iter):
                data_points = self.catmull(data_points)
            data_points = np.concatenate((data_points, np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
            num_faces = len(data_points)*2
            
            data = np.zeros(num_faces-2, dtype=mesh.Mesh.dtype)
            count = 0
            for i in range(0,num_faces-1,2):
                next_count = count + 1
                if next_count == len(data_points):
                    break
                data['vectors'][i] = np.array([data_points[count],
                                               data_points[next_count],
                                               data_points[count]+thickness])
                data['vectors'][i+1] = np.array([data_points[count]+thickness,
                                               data_points[next_count]+thickness,
                                               data_points[next_count]])
                count += 1
            # combine faces            
            all_data.append(data)
        all_data = np.concatenate(all_data)
        # generate mesh
        m = mesh.Mesh(all_data)
        # debug
        if draw:
            self.draw_stl_from_mesh(m)    
        if file_directory is None:
            if file_name is None:
                m.save('test.stl')
            else:
                m.save(file_name)
        else:
            
            with open(file_directory + file_name, 'wb') as output_file:
                m.save(file_directory + file_name, output_file, mode=stl.ASCII)



    def subdc_to_points(self, individuals, n_iter, thickness=np.array([0,0,1]),
                            file_directory=None, file_name=None, draw=False):
        """
        A method to save multiple subdivision curves in a single STL file. This may be 
        used for multiple control regions.
        
        Args:
            individuals (numpy array): DEAP control point individuals.
            n_iter (int): Number of subdivision iterations. Higher number will 
                            produce a smoother curve.
        
        Kwargs:
            thickness (numpy array): The thickness along z-axis (last element in the 
                                        array). It can also be used to offset 
                                        x- and y-axis.
            file_directory (str): The destination directory for the STL file. 
            file_name (str): The name of the STL file. 
            draw (bool): Control for drawing graphs.
        """
#        all_data = [] 
        # generate faces for each individual curve in the DEAP inidividuals
        for control_points in individuals:
            data_points = self.catmull(control_points)
            for i in range(n_iter):
                data_points = self.catmull(data_points)
                
        return data_points
    
    
    
                
                
                
#            data_points = np.concatenate((data_points, np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
#            num_faces = len(data_points)*2
#            
#            data = np.zeros(num_faces-2, dtype=mesh.Mesh.dtype)
#            count = 0
#            for i in range(0,num_faces-1,2):
#                next_count = count + 1
#                if next_count == len(data_points):
#                    break
#                data['vectors'][i] = np.array([data_points[count],
#                                               data_points[next_count],
#                                               data_points[count]+thickness])
#                data['vectors'][i+1] = np.array([data_points[count]+thickness,
#                                               data_points[next_count]+thickness,
#                                               data_points[next_count]])
#                count += 1
#            # combine faces            
#            all_data.append(data)
#        all_data = np.concatenate(all_data)
#        # generate mesh
#        m = mesh.Mesh(all_data)
#        # debug
#        if draw:
#            self.draw_stl_from_mesh(m)    
#        if file_directory is None:
#            if file_name is None:
#                m.save('test.stl')
#            else:
#                m.save(file_name)
#        else:
#            
#            with open(file_directory + file_name, 'wb') as output_file:
#                m.save(file_directory + file_name, output_file, mode=stl.ASCII)




















    def subdc_to_stl_mult_offset(self, individuals, n_iter, offset_x, offset_y, thickness=np.array([0,0,1]),
                            file_directory=None, file_name=None, draw=False):
        """
        A method to save multiple subdivision curves in a single STL file. This may be 
        used for multiple control regions.
        
        Args:
            individuals (numpy array): DEAP control point individuals.
            n_iter (int): Number of subdivision iterations. Higher number will 
                            produce a smoother curve.
        
        Kwargs:
            thickness (numpy array): The thickness along z-axis (last element in the 
                                        array). It can also be used to offset 
                                        x- and y-axis.
            file_directory (str): The destination directory for the STL file. 
            file_name (str): The name of the STL file. 
            draw (bool): Control for drawing graphs.
        """
        all_data = [] 
        # generate faces for each individual curve in the DEAP inidividuals
        for control_points in individuals:
            data_points = self.catmull(control_points)
            for i in range(n_iter):
                data_points = self.catmull(data_points)
                
            data_points = data_points + offset_x
            data_points = data_points + offset_y
                
            data_points = np.concatenate((data_points, np.zeros((len(data_points),1))-(thickness[-1]/2.0)), axis=1)
            num_faces = len(data_points)*2
            
            data = np.zeros(num_faces-2, dtype=mesh.Mesh.dtype)
            count = 0
            for i in range(0,num_faces-1,2):
                next_count = count + 1
                if next_count == len(data_points):
                    break
                data['vectors'][i] = np.array([data_points[count],
                                               data_points[next_count],
                                               data_points[count]+thickness])
                data['vectors'][i+1] = np.array([data_points[count]+thickness,
                                               data_points[next_count]+thickness,
                                               data_points[next_count]])
                count += 1
            # combine faces            
            all_data.append(data)
        all_data = np.concatenate(all_data)
        # generate mesh
        m = mesh.Mesh(all_data)
        # debug
        if draw:
            self.draw_stl_from_mesh(m)    
        if file_directory is None:
            if file_name is None:
                m.save('test.stl')
            else:
                m.save(file_name)
        else:
            
            with open(file_directory + file_name, 'wb') as output_file:
                m.save(file_directory + file_name, output_file, mode=stl.ASCII)
    

    
    def catmull(self, P):
        """
        A method to generate subdivision curves with given control points.
    
        Args:
            P (numpy array): control points.
    
        Returns:
            Q (numpy array): generated points on the subdivision curve.
        """
        N = P.shape[0]
        Q = np.zeros((2*N-1, 2), 'd')
        Q[0,:] = P[0,:]
        for i in range(0,N-1):
            if i > 0:
                Q[2*i,:] = (P[i-1,:]+6*P[i,:]+P[i+1,:])/8
            Q[2*i+1,:] = (P[i,:]+P[i+1,:])/2
        Q[-1,:] = P[-1,:]
        return Q

    def draw_stl_from_mesh(self, m):
        """
        A method to draw numpy-stl mesh.
    
        Args:
            m (numpy-stl mesh): mesh object.
    
        """
        plt.ion()
        # Create a new plot
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)
    
        # Render the cube faces

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
        
        axes.view_init(30, 210)
        # Auto scale to the mesh size
        
        axes.set_xlim3d(-1200, 0)
        axes.set_ylim3d( 0, 800)
        axes.set_zlim3d(-100, 100)
        axes.tick_params(axis='x',pad=10)
        axes.tick_params(axis='y',pad=10)
        axes.tick_params(axis='z',pad=20)
        axes.set_xlabel(r'$x \ (mm)$', labelpad=40)
        axes.set_ylabel(r'$y \ (mm)$', labelpad=40)
        axes.set_zlabel(r'$z \ (mm)$', labelpad=40)
        plt.savefig("stl.png")

    def baffle_constraint_check(self, x, duct_boundary):
        shape = self.convert_dec2pos(x) # convert random array to ordered list
        points = [shape]
        for control_points in points:
            data_points = self.catmull(control_points)
            for j in range(self.niterations):
                data_points = self.catmull(data_points)
        
        tree = spatial.cKDTree(data_points)
        
        mindist, minid = tree.query(duct_boundary, n_jobs=16)
        minimum_distance = min(mindist)
        
        if(minimum_distance >= self.duct_constraint):
            return True
        else:
            return False


    def duct_constraint_check(self, x, corner_boundary, baffle_boundary):
        shape = self.convert_dec2pos(x) # convert random array to ordered list
        points = [shape]
        for control_points in points:
            data_points = self.catmull(control_points)
            for j in range(self.niterations):
                data_points = self.catmull(data_points)
        
        tree = spatial.cKDTree(data_points)
        
        mindist_corner, minid_corner = tree.query(corner_boundary, n_jobs=16)
        minimum_distance_corner = min(mindist_corner)
        
        mindist_baffle, minid_baffle = tree.query(baffle_boundary, n_jobs=16)
        minimum_distance_baffle = min(mindist_baffle)        
        
        if(minimum_distance_corner >= self.corner_constraint) and (minimum_distance_baffle >= self.baffle_constraint):
            return True
        else:
            return False






    def constraint_boundaries_only(self, x):
        #pdb.set_trace()
        pos = np.reshape(x, (-1, 2))
        pts_comp = [self.polygon.contains_points([pts]) for pts in pos]
        if np.all(pts_comp):
            return True            
        return False

    def get_dec_boundary_duct(self):
        lb = np.ones(self.vertices.shape) * np.min(self.vertices[1:4], axis=0)
        mlb = self.convert_pos2dec(lb)
        ub = np.ones(self.vertices.shape) * np.max(self.vertices[1:4], axis=0)
        mub = self.convert_pos2dec(ub)
        return mlb, mub

    def get_dec_boundary_baffles(self):
        lb = np.ones(self.vertices.shape) * np.min(self.vertices[1:4], axis=0)   # was [6:10]
        mlb = self.convert_pos2dec(lb)
        ub = np.ones(self.vertices.shape) * np.max(self.vertices[1:4], axis=0)
        mub = self.convert_pos2dec(ub)
        return mlb, mub
    
    def lhs_initial_samples(self, n_dim, ub, lb, seed, n_samples=4, cfunc=None, cargs=(), ckwargs={}):
        """
        Generate Latin hypercube samples from the decision space using pyDOE.
    
        Parameters.
        -----------
        n_samples (int): the number of samples to take. 
        cfunc (method): a cheap constraint function.
        cargs (tuple): arguments for cheap constraint function.
        ckawargs (dictionary): keyword arguments for cheap constraint function. 
    
        Returns a set of decision vectors.         
        """
        #seed = 1435 # was 1435
        np.random.seed(seed)
        samples = LHS(n_dim, samples=n_samples)
        
        scaled_samples_unconstrained = ((ub - lb) * samples) + lb            
            
        if cfunc is not None: # check for constraints
            print('Checking for constraints.')
            scaled_samples = np.array([i for i in scaled_samples_unconstrained if cfunc(i, *cargs, **ckwargs)])
    
            return scaled_samples, scaled_samples_unconstrained
        
        else:
            
            return scaled_samples_unconstrained, scaled_samples_unconstrained

#
#from turbulucid import *
#import matplotlib
#matplotlib.rcParams['figure.figsize'] = (20, 20)
#matplotlib.rcParams['axes.labelsize'] = 40
#matplotlib.rcParams['xtick.labelsize'] = 40
#matplotlib.rcParams['ytick.labelsize'] = 40
#matplotlib.rcParams['text.usetex'] = True
#import turbulucid
#import importlib
#importlib.reload(turbulucid) # have to reload module if changes are made
#
#class Plotting(object):
#
#    def __init__(self, case, corner_boundary, baffle_boundary, duct_boundary, fixed_points, decision_boundary_duct, decision_boundary_baffles, decision_boundary_corner, choose_decision_number):
#
#        plot_boundaries(case)
#        self.corner_boundary = corner_boundary
#        self.baffle_boundary = baffle_boundary
#        self.duct_boundary = duct_boundary
#        self.decision_boundary_duct = decision_boundary_duct
#        self.decision_boundary_baffles = decision_boundary_baffles
#        self.decision_boundary_corner = decision_boundary_corner
#        self.fixed_points = fixed_points
#        self.choose_decision_number = choose_decision_number
#        
#    def plot_corner_boundary(self):
#        
#        # corner points
#        x_corner = []
#        y_corner = []
#        
#        for k in range(len(self.corner_boundary)):
#            x_corner.append(self.corner_boundary[k][0])
#            y_corner.append(self.corner_boundary[k][1])
#        
#        plt.plot(x_corner,y_corner,'r-', markersize=2.5, linewidth=4)
#        
#    def plot_baffle_boundary(self):
#        
#        # baffle points
#        x_baffle = []
#        y_baffle = []
#        
#        for k in range(len(self.baffle_boundary)):
#            x_baffle.append(self.baffle_boundary[k][0])
#            y_baffle.append(self.baffle_boundary[k][1])
#        
#        
#        length=74
#        
#        plt.plot(x_baffle[:length],y_baffle[:length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[length:2*length],y_baffle[length:2*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[2*length:3*length],y_baffle[2*length:3*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[3*length:4*length],y_baffle[3*length:4*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[4*length:5*length],y_baffle[4*length:5*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[5*length:6*length],y_baffle[5*length:6*length],'r-', markersize=2.5, linewidth=4)
#        
#        plt.plot(x_baffle[6*length:7*length],y_baffle[6*length:7*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[7*length:8*length],y_baffle[7*length:8*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[8*length:9*length],y_baffle[8*length:9*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[9*length:10*length],y_baffle[9*length:10*length],'r-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[10*length:11*length],y_baffle[10*length:11*length],'r-', markersize=2.5, linewidth=4)        
#
#
#    # 125 micron correction
#    def plot_duct_boundary(self):
#        
#        
#                # corner points
#        x_duct = []
#        y_duct = []
#        
#        for k in range(len(self.duct_boundary)):
#            x_duct.append(self.duct_boundary[k][0])
#            y_duct.append(self.duct_boundary[k][1])
#        
#        plt.plot(x_duct,y_duct,'r-', markersize=2.5, linewidth=4)
#        
#        
#        
#        
##        # linear part of baffle
##        startx = -34.851
##        stopx = -17.250
##        x_duct_linear = np.linspace(startx, stopx, num=20, endpoint=True)[:,None]
##
##        starty = 47.770
##        stopy = 19.668
##        y_duct_linear = np.linspace(starty, stopy, num=20, endpoint=True)[:,None]
##        
##        plt.plot(x_duct_linear, y_duct_linear, color='magenta', linestyle='-', markersize=2.5, linewidth=4)
##
#
#
#
#        
#    def plot_decision_boundary_and_fixed_points(self):
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_duct)-5):
#            x_boundary.append(self.decision_boundary_duct[i][0])
#            y_boundary.append(self.decision_boundary_duct[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_duct)-5, len(self.decision_boundary_duct)-1):
#            x_boundary.append(self.decision_boundary_duct[i][0])
#            y_boundary.append(self.decision_boundary_duct[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)        
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_baffles)-5):
#            x_boundary.append(self.decision_boundary_baffles[i][0])
#            y_boundary.append(self.decision_boundary_baffles[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_baffles)-5, len(self.decision_boundary_baffles)-1):
#            x_boundary.append(self.decision_boundary_baffles[i][0])
#            y_boundary.append(self.decision_boundary_baffles[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)
#
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_corner)-5):
#            x_boundary.append(self.decision_boundary_corner[i][0])
#            y_boundary.append(self.decision_boundary_corner[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(self.decision_boundary_corner)-5, len(self.decision_boundary_corner)-1):
#            x_boundary.append(self.decision_boundary_corner[i][0])
#            y_boundary.append(self.decision_boundary_corner[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=12, linewidth=4)
#
#        # fixed_points
#        fixed=[self.fixed_points[0][0],self.fixed_points[0][1],self.fixed_points[1][0],self.fixed_points[1][1]]
#        
#        x_fixed = []
#        y_fixed = []
#        
#        for j in fixed:
#            x_fixed.append(self.decision_boundary_duct[j][0])
#            y_fixed.append(self.decision_boundary_duct[j][1])
#        
#        plt.plot(x_fixed,y_fixed,'ks', markersize=12)
#        
#        x_fixed = []
#        y_fixed = []
#        
#        for j in fixed:
#            x_fixed.append(self.decision_boundary_baffles[j][0])
#            y_fixed.append(self.decision_boundary_baffles[j][1])
#        
#        plt.plot(x_fixed,y_fixed,'ks', markersize=12)   
#        
#        x_fixed = []
#        y_fixed = []
#        
#        for j in fixed:
#            x_fixed.append(self.decision_boundary_corner[j][0])
#            y_fixed.append(self.decision_boundary_corner[j][1])
#        
#        plt.plot(x_fixed,y_fixed,'ks', markersize=12)          
#               
##        x_c = -16.24
##        y_c = 59.211
##        r = corner_constraint
##        start=0
##        stop=2*math.pi
##        theta = np.linspace(start, stop, num=100, endpoint=True)[:,None]
##        x_radius = x_c - r * np.sin(theta)
##        y_radius = y_c - r * np.cos(theta)
##        plt.plot(x_radius, y_radius,'k-', alpha=0.25)         
#        
#        plt.xlim([-45,0])
#        plt.ylim([15,60]) # 883.7 offset because of inlet
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        plt.tight_layout()
#        plt.savefig("duct.png")
#                
#        plt.xlim([-17.5-0.5,-14+0.5])
#        plt.ylim([19.6-0.5,23.1+0.5])
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        plt.tight_layout()
#        plt.savefig("baffle.png")
#        
#        plt.xlim([-17.5-0.5,-14+0.5])
#        plt.ylim([19.6-0.5+11*3.438,23.1+0.5+11*3.438])
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        plt.tight_layout()
#        plt.savefig("corner.png")
#
#
#
#
#
#
#
#
#
#
#
#
#    def plot_decision_and_catmull_clark_curve_baffle(self, rand_x, decision_curve):
#        
#        for i in range(0,len(rand_x[self.choose_decision_number]),2):
#            plt.plot(rand_x[self.choose_decision_number][i],rand_x[self.choose_decision_number][i+1], 'r^', markersize=14,zorder=1)
#        
#        # catmull_clark curve
#        x_curve = []
#        y_curve = []
#        
#        for k in range(0, len(decision_curve)):
#            x_curve.append(decision_curve[k][0])
#            y_curve.append(decision_curve[k][1])
#    
#        plt.plot(x_curve,y_curve,'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+2.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+3.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+4.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+5.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+6.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+7.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)        
#        plt.plot(x_curve,y_curve+8.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+9.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)
#        plt.plot(x_curve,y_curve+10.0*3.438*np.ones_like(y_curve),'-', color='green', linewidth=4, zorder=10)        
#
#
#
#    def plot_decision_and_catmull_clark_curve_duct(self, rand_x, decision_curve):
#        
#        for i in range(0,len(rand_x[self.choose_decision_number]),2):
#            plt.plot(rand_x[self.choose_decision_number][i],rand_x[self.choose_decision_number][i+1], 'r^', markersize=14,zorder=1)
#        
#        # catmull_clark curve
#        x_curve = []
#        y_curve = []
#        
#        for k in range(0, len(decision_curve)):
#            x_curve.append(decision_curve[k][0])
#            y_curve.append(decision_curve[k][1])
#    
#        plt.plot(x_curve,y_curve,'-', color='green', linewidth=4, zorder=10)
#        
#    def plot_decision_and_catmull_clark_curve_corner(self, rand_x, decision_curve):
#        
#        
#        #corner_scale_factor =  (22.481365 -  21.398)   / (59.57776086400099-59.211)
#        corner_subtracted = rand_x-np.array([0, 21.398])
#        corner_scaled = corner_subtracted/np.array([1,corner_scale_factor])
#        corner = corner_scaled + np.array([0, 59.197])
#        
#        
#        for i in range(0,len(corner[self.choose_decision_number]),2):
#            plt.plot(corner[self.choose_decision_number][i],corner[self.choose_decision_number][i+1], 'r^', markersize=14,zorder=1)
#        
#        # catmull_clark curve
#        x_curve = []
#        y_curve = []
#        
#        for k in range(0, len(decision_curve)):
#            x_curve.append(decision_curve[k][0])
#            y_curve.append(decision_curve[k][1])
#    
#        plt.plot(x_curve,y_curve,'-', color='green', linewidth=4, zorder=10)
#        
#        
#        
#    def plot_all_curves(self, rand_x, rand_x_baffle):
#        
#        fig = plt.figure(figsize=(20,20))
#        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
#        ax1=fig.add_subplot(1,1,1)        
#        
#        for duct_decision in rand_x:
#        
#            shape = ductPolygon.convert_dec2pos(duct_decision) # convert random array to ordered list
#            points = [shape]
#            for control_points in points:
#                decision_curve = ductPolygon.catmull(control_points)
#                for i in range(niterations):
#                    decision_curve = ductPolygon.catmull(decision_curve)
#                    
#            x_curve = []
#            y_curve = []
#            
#            for k in range(0, len(decision_curve)):
#                x_curve.append(decision_curve[k][0])
#                y_curve.append(decision_curve[k][1])   
#                
#            ax1.plot(x_curve,y_curve,'-', linewidth=4, zorder=10)
#                    
#        for baffle_decision in rand_x_baffle:
#            shape_baffle = bafflePolygon.convert_dec2pos(baffle_decision) # convert random array to ordered list
#            points = [shape_baffle]
#            for control_points in points:
#                decision_curve_baffle = bafflePolygon.catmull(control_points)
#                for i in range(niterations):
#                    decision_curve_baffle = bafflePolygon.catmull(decision_curve_baffle) 
#                    
#            x_curve = []
#            y_curve = []
#            
#            for k in range(0, len(decision_curve_baffle)):
#                x_curve.append(decision_curve_baffle[k][0])
#                y_curve.append(decision_curve_baffle[k][1])             
#        
#            ax1.plot(x_curve,y_curve,'-', linewidth=4, zorder=10)
#            
#            
#            
#        for corner_decision in rand_x_baffle:
#            shape_corner = cornerPolygon.convert_dec2pos_corner(corner_decision) # convert random array to ordered list
#            points = [shape_corner]
#            for control_points in points:
#                decision_curve_corner = cornerPolygon.catmull(control_points)
#                for i in range(niterations):
#                    decision_curve_corner = cornerPolygon.catmull(decision_curve_corner) 
#                    
#            x_curve = []
#            y_curve = []
#            
#            for k in range(0, len(decision_curve_corner)):
#                x_curve.append(decision_curve_corner[k][0])
#                y_curve.append(decision_curve_corner[k][1])             
#        
#            ax1.plot(x_curve,y_curve,'-', linewidth=4, zorder=10)            
#            
#            
#            
#            
#            
#            
#        x_duct = []
#        y_duct = []
#        
#        for k in range(len(plot.duct_boundary)):
#            x_duct.append(plot.duct_boundary[k][0])
#            y_duct.append(plot.duct_boundary[k][1])
#        
#        ax1.plot(x_duct,y_duct,'k-', markersize=2.5, linewidth=4, zorder=20)    
#            
#        # baffle points
#        x_baffle = []
#        y_baffle = []
#        
#        for k in range(len(plot.baffle_boundary)):
#            x_baffle.append(plot.baffle_boundary[k][0])
#            y_baffle.append(plot.baffle_boundary[k][1])
#        
#        ax1.plot(x_baffle[:74],y_baffle[:74],'k-', markersize=2.5, linewidth=4, zorder=20)
#        
#        plot_boundaries(case)    
#        
#        
#        plt.xlim([-17.5-0.5,-14+0.5])
#        plt.ylim([19.6-0.5,23.1+0.5])
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        
#        
#        
#        
##        ax1.set_xlim([-420, -290])
##        ax1.set_ylim([(100-883.7), (230-883.7)])
##        ax1.set_xlabel(r'$x \ (mm)$')
##        ax1.set_ylabel(r'$y \ (mm)$')
#        plt.tight_layout()
#        plt.savefig("all_curves_baffle.png")
#            
#
#
#        # baffle points
#        x_corner = []
#        y_corner = []
#        
#        for k in range(len(plot.corner_boundary)):
#            x_corner.append(plot.corner_boundary[k][0])
#            y_corner.append(plot.corner_boundary[k][1])
#        
#        ax1.plot(x_corner,y_corner,'k-', markersize=2.5, linewidth=4, zorder=20)
#
#        
#        
#        plt.xlim([-17.5-0.5,-14+0.5])
#        plt.ylim([19.6-0.5+11*3.438,23.1+0.5+11*3.438])
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        
#        plt.tight_layout()
#        plt.savefig("all_curves_corner.png")
#
#
#
#
#
#        length=74
#
#        plt.plot(x_baffle[:length],y_baffle[:length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[length:2*length],y_baffle[length:2*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[2*length:3*length],y_baffle[2*length:3*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[3*length:4*length],y_baffle[3*length:4*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[4*length:5*length],y_baffle[4*length:5*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[5*length:6*length],y_baffle[5*length:6*length],'k-', markersize=2.5, linewidth=4)
#        
#        plt.plot(x_baffle[6*length:7*length],y_baffle[6*length:7*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[7*length:8*length],y_baffle[7*length:8*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[8*length:9*length],y_baffle[8*length:9*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[9*length:10*length],y_baffle[9*length:10*length],'k-', markersize=2.5, linewidth=4)
#        plt.plot(x_baffle[10*length:11*length],y_baffle[10*length:11*length],'k-', markersize=2.5, linewidth=4)  
#        
#        
#        plt.xlim([-45,0])
#        plt.ylim([15,60]) # 883.7 offset because of inlet
#        plt.xlabel(r'$x \ (in)$')
#        plt.ylabel(r'$y \ (in)$')
#        
#        
#        plt.tight_layout()
#        plt.savefig("all_curves_duct.png")
#
#    
#    def plot_mesh(self, filename):
#        import vtk
#        from numpy import zeros
#        import matplotlib.pyplot as plt
#        #import matplotlib
#        
#        reader = vtk.vtkUnstructuredGridReader()
#        reader.SetFileName(filename)
#        reader.Update()
#        
#        plane = vtk.vtkPlane()
#        plane.SetOrigin(0, 0, 0)
#        plane.SetNormal(0, 0, 1)
#        
#        #cutter = vtk.vtkFiltersCorePython.vtkCutter()
#        cutter = vtk.vtkCutter()
#        cutter.SetCutFunction(plane)
#        cutter.SetInputConnection(reader.GetOutputPort())
#        cutter.Update()
#        
#        data = cutter.GetOutput()
#        
#        triangles = data.GetPolys().GetData()
#        points = data.GetPoints()
#        
#        mapper = vtk.vtkCellDataToPointData()
#        mapper.AddInputData(data)
#        mapper.Update()
#        
#        ntri = triangles.GetNumberOfTuples()/4
#        npts = points.GetNumberOfPoints()
#        
#        tri = zeros((int(ntri), 3))
#        x = zeros(npts)
#        y = zeros(npts)
#        
#        for i in range(0, int(ntri)):
#            tri[i, 0] = triangles.GetTuple(4*i + 1)[0]
#            tri[i, 1] = triangles.GetTuple(4*i + 2)[0]
#            tri[i, 2] = triangles.GetTuple(4*i + 3)[0]
#            
#        for i in range(npts):
#            pt = points.GetPoint(i)
#            x[i] = pt[0]
#            y[i] = pt[1]
#        
#        
#        # Mesh
#        
#        #matplotlib.rcParams.update({'font.size': 18})
#            
#        fig = plt.figure(figsize=(20,20))
#        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
#        ax1=fig.add_subplot(1,1,1)     
#            
#        ax1.triplot(x, y, tri,'k-', lw=0.5)
#        ax1.set_xlim([-420, -290])
#        ax1.set_ylim([(100-883.7), (230-883.7)])
#        ax1.set_xlabel(r'$x \ (mm)$')
#        ax1.set_ylabel(r'$y \ (mm)$')
#        plt.savefig("mesh.png")
#        
#
#
#    def plot_lhs_duct_points(self):
#
##        fig = plt.figure(figsize=(40,40))
##        SMALL_SIZE = 48
##        plt.rc('text', usetex=True)
##        plt.rc('font', family='serif') 
##        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
##        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
##        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
##        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
##        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
##        
##        
##        xlines = np.linspace(lb_duct[0], ub_duct[0],n_samples_duct+1)
##        ylines = np.linspace(lb_duct[1], ub_duct[1],n_samples_duct+1)
##        
##        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
##        ax1=fig.add_subplot(1,1,1)
##        for i in range(len(xlines)):     
##            ax1.plot(xlines[i]*np.ones(len(xlines)), np.ones(len(xlines))*ylines, color='gray', alpha=0.5 )
##            
##        for j in range(len(xlines)):     
##            ax1.plot(xlines*np.ones(len(xlines)), np.ones(len(xlines))*ylines[j], color='gray', alpha=0.5 )
##              
##        ax1.plot(lhs_duct[:,0], lhs_duct[:,1],'rx', lw=1, markersize=15,markeredgewidth=2, label='Point 1')
##        ax1.set_xlim([-753.65, -406.0])
##        ax1.set_ylim([-761.0, -250.0])
##        ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
##        plt.tight_layout()
##        ax1.set_xlabel(r'$x \ (mm)$')
##        ax1.set_ylabel(r'$y \ (mm)$')
##        plt.savefig("lhs_duct.png")
#
#
#        fig = plt.figure(figsize=(40,40))
#        SMALL_SIZE = 90
#        plt.rc('text', usetex=True)
#        plt.rc('font', family='serif') 
#        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
#        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
#        xlines = np.linspace(lb_duct[0], ub_duct[0], n_samples_duct+1)
#        ylines = np.linspace(lb_duct[1], ub_duct[1], n_samples_duct+1)
#        
#        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
#        ax1=fig.add_subplot(1,1,1)
#        for i in range(len(xlines)):     
#            ax1.plot(xlines[i]*np.ones(len(xlines)), np.ones(len(xlines))*ylines, color='black', alpha=1 )
#            
#        for j in range(len(xlines)):     
#            ax1.plot(xlines*np.ones(len(xlines)), np.ones(len(xlines))*ylines[j], color='black', alpha=1 )
#              
#            
#            
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(decision_boundary_duct)-5):
#            x_boundary.append(decision_boundary_duct[i][0])
#            y_boundary.append(decision_boundary_duct[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=24, linewidth=4)
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(decision_boundary_duct)-5, len(decision_boundary_duct)-1):
#            x_boundary.append(decision_boundary_duct[i][0])
#            y_boundary.append(decision_boundary_duct[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=24, linewidth=4)
#            
#            
#            
#            
#            
#            
#        ax1.plot(lhs_duct[:,0], lhs_duct[:,1],'ro', lw=1, markersize=30,markeredgewidth=4, label='Point 1')
##            ax1.set_xlim([-390.74, -311.44])
##            ax1.set_ylim([-730.32, -701.68])
#        #ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
#        plt.tight_layout()
#        ax1.set_xlabel(r'$x \ (in)$')
#        ax1.set_ylabel(r'$y \ (in)$')
#        plt.savefig("lhs_duct.png")
#
#
#
#
#
#
#    def plot_lhs_baffle_points(self):
#
##        fig = plt.figure(figsize=(40,40))
##        SMALL_SIZE = 48
##        plt.rc('text', usetex=True)
##        plt.rc('font', family='serif') 
##        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
##        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
##        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
##        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
##        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
##        
##        
##        xlines = np.linspace(lb_baffle[0], ub_baffle[0], n_samples_baffle+1)
##        ylines = np.linspace(lb_baffle[1], ub_baffle[1], n_samples_baffle+1)
##        
##        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
##        ax1=fig.add_subplot(1,1,1)
##        for i in range(len(xlines)):     
##            ax1.plot(xlines[i]*np.ones(len(xlines)), np.ones(len(xlines))*ylines, color='gray', alpha=0.5 )
##            
##        for j in range(len(xlines)):     
##            ax1.plot(xlines*np.ones(len(xlines)), np.ones(len(xlines))*ylines[j], color='gray', alpha=0.5 )
##              
##        ax1.plot(lhs_baffle[:,0], lhs_baffle[:,1],'rx', lw=1, markersize=15,markeredgewidth=2, label='Point 1')
##        ax1.set_xlim([-390.74, -311.44])
##        ax1.set_ylim([-730.32, -701.68])
##        ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
##        plt.tight_layout()
##        ax1.set_xlabel(r'$x \ (mm)$')
##        ax1.set_ylabel(r'$y \ (mm)$')
##        plt.savefig("lhs_baffle.png")
#        
#        
#        
#        
#        fig = plt.figure(figsize=(40,40))
#        SMALL_SIZE = 90
#        plt.rc('text', usetex=True)
#        plt.rc('font', family='serif') 
#        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#        plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
#        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#        plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
#        xlines = np.linspace(lb_baffle[0], ub_baffle[0], n_samples_baffle+1)
#        ylines = np.linspace(lb_baffle[1], ub_baffle[1], n_samples_baffle+1)
#        
#        fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
#        ax1=fig.add_subplot(1,1,1)
#        for i in range(len(xlines)):     
#            ax1.plot(xlines[i]*np.ones(len(xlines)), np.ones(len(xlines))*ylines, color='black', alpha=1 )
#            
#        for j in range(len(xlines)):     
#            ax1.plot(xlines*np.ones(len(xlines)), np.ones(len(xlines))*ylines[j], color='black', alpha=1 )
#              
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(decision_boundary_baffles)-5):
#            x_boundary.append(decision_boundary_baffles[i][0])
#            y_boundary.append(decision_boundary_baffles[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=24, linewidth=4)
#        
#        # boundary points
#        x_boundary = []
#        y_boundary = []
#        
#        for i in range(len(decision_boundary_baffles)-5, len(decision_boundary_baffles)-1):
#            x_boundary.append(decision_boundary_baffles[i][0])
#            y_boundary.append(decision_boundary_baffles[i][1])
#        
#        plt.plot(x_boundary,y_boundary,'bo--', markersize=24, linewidth=4)
#
# 
#            
#            
#        ax1.plot(lhs_baffle[:,0], lhs_baffle[:,1],'ro', lw=1, markersize=30,markeredgewidth=4, label='Point 1')
##            ax1.set_xlim([-390.74, -311.44])
##            ax1.set_ylim([-730.32, -701.68])
#        #ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
#        plt.tight_layout()
#        ax1.set_xlabel(r'$x \ (in)$')
#        ax1.set_ylabel(r'$y \ (in)$')
#        plt.savefig("lhs_baffle.png")
#        
#        
#        
#        
#        
#        
#        
#        
        
        
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

#    if((f[0] >= 0.6) and (f[0] <= 0.8)):
#        return [0, 0]
#    else:
    return f        
        
        
        


###############################################################################
## MAIN FUNCTION
###############################################################################

if __name__ == '__main__':

    ###############
    # PRELIMINARY #
    ###############
    
#    # Start time
    start_sim = time.time()
#    
#    function='openfoam'
#    
#    current = os.getcwd()
#        
#    # Seed       
##    seed = 1005
##    np.random.seed(seed)
#    
#    case_path = '/data/HeadCell/case/'
#    source_path = '/data/HeadCell/source/'
#    completed_path = '/data/HeadCell/completed/'
#    
#    # Sets values and removes olds case directory
#    subprocess.call(['rm', '-r', current + case_path])
#    
#    subprocess.call(['mkdir', '-p', current + case_path])
#    subprocess.call(['mkdir', '-p', current + completed_path])
#    
#    print("main(): removed old case, copied new case and created new mesh directories")    
#
#    ###############
#    # PRELIMINARY #
#    ###############    
#    
#    ##########
#    # INPUTS #
#    ##########
#    
#    n_control = 1
#    niterations = 5    
#    
#    # computation for baffle_points - used in csv file not for code
#    m = (( 22.501285 - 21.476099 ) / ( -16.942134 - -16.299834))
#    baffle_point = (((22.501285-0.25)-22.501285)/m)-16.942134
#    #baffle_point_2 = 21.398-(22.231365-21.398)
#    corner_point = (( (59.840578 - 59.197 ))/((-17.25 - - 16.239)))*(baffle_point - -17.25) + (59.840578)
#    
#    
#    
#    
#    # Needed for scaling corner point in code:
#    # scaling occurs by taking the ratio of the difference between the maxima and minima of each decision space
#    corner_point_2 = (( (59.840578 - 59.197 ))/((-17.25 - - 16.239)))*(-16.942134 - -17.25) + (59.840578)
#    corner_scale_factor =  (22.481365 -  21.398)   / (corner_point_2 - 59.197)
#    #corner_point = (59.211+0.005) - (59.49312618190835-(59.211+0.005))
#    
#    
#    
#    
#    
#    
#    
#    
#    baffle_constraint = 0*1.711               # 125 micron correction NOT ACTIVE
#    corner_constraint = 0*20.608 #17.895 #20.608     # 125 micron correction
#    duct_constraint = 0*(1.711+0.04)
#    
#    n_dim = 2
#    n_samples_duct = 43 # there are 4 dimensions - 2 per domain - 2 domains
#    n_samples_baffle = 43
#    seed_duct = 52822
#    seed_baffle = 52822
#    
#    fixed_points = list(np.loadtxt('fixed_points.csv', delimiter=',').astype(int))
#    decision_boundary_duct = np.loadtxt('decision_boundary_duct.csv', delimiter=',')                # 125 micron correction
#    decision_boundary_baffles = np.loadtxt('decision_boundary_baffles.csv', delimiter=',')
#    decision_boundary_corner = np.loadtxt('decision_boundary_corner.csv', delimiter=',')
#
#
#    ##########
#    # INPUTS #
#    ##########
#    
#    #########################
#    # CONSTRAINT BOUNDARIES #
#    #########################
#    
#    ductPolygon = ControlPolygon2D(decision_boundary_duct, fixed_points, n_control, corner_constraint, baffle_constraint, duct_constraint, niterations)
#    duct_boundary = ductPolygon.create_duct_boundary_constraint()
#    
#    bafflePolygon = ControlPolygon2D(decision_boundary_baffles, fixed_points, n_control, corner_constraint, baffle_constraint, duct_constraint, niterations)
#    baffle_boundary = bafflePolygon.create_baffle_boundary_constraint()
#
#    cornerPolygon = ControlPolygon2D(decision_boundary_corner, fixed_points, n_control, corner_constraint, baffle_constraint, duct_constraint, niterations)
#    corner_boundary = bafflePolygon.create_corner_boundary_constraint()
#
#    lb_duct, ub_duct = ductPolygon.get_dec_boundary_duct()
#
#    lb_baffle, ub_baffle = bafflePolygon.get_dec_boundary_baffles()
#    
#    
#    
#    
#    
#    ub = np.concatenate((ub_duct, ub_baffle))
#    lb = np.concatenate((lb_duct, lb_baffle))

    #########################
    # CONSTRAINT BOUNDARIES #
    #########################
    
    ##############
    # CREATE LHS #
    ##############
    
    # generate random solutions within the bounds satisfying the constraints

#    lhs_duct, all_duct = ductPolygon.lhs_initial_samples(n_dim, ub_duct, lb_duct, seed_duct, n_samples_duct, cfunc=None, cargs=(), ckwargs={})
#    print("initialisation(): number of Latin Hypercube samples", str(n_samples_duct))
#    print('initialisation(): number of samples that pass constraints, ', len(lhs_duct))
#    print('initialisation(): number of samples unconstrained, ', len(all_duct))
#    
#    lhs_baffle, all_baffle = bafflePolygon.lhs_initial_samples(n_dim, ub_baffle, lb_baffle, seed_baffle, n_samples_baffle, cfunc=None, cargs=(), ckwargs={})
#    print("initialisation(): number of Latin Hypercube samples", str(n_samples_baffle))
#    print('initialisation(): number of samples that pass constraints, ', len(lhs_baffle))
#    print('initialisation(): number of samples unconstrained, ', len(all_baffle))
#
#    lhs_samples = np.append(lhs_duct, lhs_baffle, axis=1)

    ##############
    # CREATE LHS #
    ##############






    ###############
    # LHS OPTIONS #
    ###############
    
#    monotonic_euclidean_distance_duct = 0.0
#    monotonic_euclidean_distance_baffle = 0.0
#    
#    metric_1_seed_samples_duct = []
#    metric_2_distances_duct = []
#    
#    metric_1_seed_samples_baffle = []
#    metric_2_distances_baffle = []    
#    
#    
#    print('Writing empty LHS_options.npz file...')
#            
#    # the name of the sim_file is initial_samples.npz    
#    sim_file_duct = 'LHS_options_duct.npz'
#    
#    # remove npz file if it exists
#    subprocess.call(['rm', '-r', current + '/' + sim_file_duct]) 
#    
#    try:
#        np.savez(current + '/' + sim_file_duct, metric_1_seed_samples_duct, metric_2_distances_duct)
#        print('Data saved in file: ', sim_file_duct)
#    except Exception as e:
#        print(e)
#        print('Data saving failed.')
#        
#        
#        
#        
#    
#    from scipy.spatial import distance
#
#    initial = 15611
#    
#    for seed_duct in range(initial, initial+30000):
#    
#        lhs_duct, all_duct = ductPolygon.lhs_initial_samples(n_dim, ub_duct, lb_duct, seed_duct, n_samples_duct, cfunc=ductPolygon.duct_constraint_check, cargs=(corner_boundary,baffle_boundary,), ckwargs={})
#        print("initialisation(): number of Latin Hypercube samples", str(n_samples_duct))
#        print('initialisation(): number of samples that pass constraints, ', len(lhs_duct))
#        print('initialisation(): number of samples unconstrained, ', len(all_duct))    
#
#        if(len(lhs_duct) == 43):
#            
#            dists_duct = distance.pdist((lhs_duct - lb_duct)/(ub_duct - lb_duct), metric='euclidean')
#            
#            min_euclidean_distance_duct = min(dists_duct)
#    
#            if (min_euclidean_distance_duct > monotonic_euclidean_distance_duct):
#                monotonic_euclidean_distance_duct = min_euclidean_distance_duct
#                
#                print("monotonic_euclidean_distance_duct: ", monotonic_euclidean_distance_duct)
#                
#                metric_1_seed_samples_duct.append(seed_duct)
#                metric_2_distances_duct.append(monotonic_euclidean_distance_duct)
#    
##                fig = plt.figure(figsize=(40,40))
##                SMALL_SIZE = 48
##                plt.rc('text', usetex=False)
##                plt.rc('font', family='serif') 
##                plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
##                plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
##                plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
##                plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##                plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
##                plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
##                plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
##                xlines = np.linspace(lb_baffle[0], ub_baffle[0], n_samples_baffle+1)
##                ylines = np.linspace(lb_baffle[1], ub_baffle[1], n_samples_baffle+1)
##                
##                fig.subplots_adjust(bottom=0.060, top=0.985, left=0.060, right=0.985)        
##                ax1=fig.add_subplot(1,1,1)
##                for i in range(len(xlines)):     
##                    ax1.plot(xlines[i]*np.ones(len(xlines)), np.ones(len(xlines))*ylines, color='gray', alpha=0.5 )
##                    
##                for j in range(len(xlines)):     
##                    ax1.plot(xlines*np.ones(len(xlines)), np.ones(len(xlines))*ylines[j], color='gray', alpha=0.5 )
##                      
##                ax1.plot(lhs_baffle[:,0], lhs_baffle[:,1],'rx', lw=1, markersize=15,markeredgewidth=2, label='Point 1')
##                ax1.set_xlim([-390.74, -311.44])
##                ax1.set_ylim([-730.32, -701.68])
##                ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
##                plt.tight_layout()
##                ax1.set_xlabel(r'$x \ (mm)$')
##                ax1.set_ylabel(r'$y \ (mm)$')
##                plt.savefig("lhs_baffle_"+str(seed_baffle)+".png")
#    
#    
#                try:
#                    np.savez(current + '/' + sim_file_duct, metric_1_seed_samples_duct, metric_2_distances_duct)
#                    print('Data saved in file: ', sim_file_duct)
#                except Exception as e:
#                    print(e)
#                    print('Data saving failed.')

    ###############
    # LHS OPTIONS #
    ###############
























    
    


    ######################
    # CREATE DIRECTORIES #
    ######################

#    # create filedirs
#    filedirs = []
#
#    stl_dir = "/constant/triSurface/"
#    
#    # loop through the list to create directories:
#    #if(lhs_samples[0][0] != None):
#    for s in lhs_samples:
#            
#        #s = lhs_samples[0]
#        
#        # create a working directory from the sample:
#        dir_name = "_"
#        for j in range(len(s)):
#            dir_name += "{0:.3f}_".format(s[j])
#        
#        # replace any directories containing []
#        dir_name = dir_name.replace("[","")
#        dir_name = dir_name.replace("]","")
#        
#        # add the name to a list of directories
#        filedirs.append(current + case_path + dir_name)
#        
#        # create the directory from the last in the list and 
#        subprocess.call(['mkdir', filedirs[-1] + '/'])
#        
#        # copy all source files into the newly created directory
#        subprocess.call(['cp', '-r', current + source_path + '.', filedirs[-1]])
#    
#        with open(filedirs[-1] + "/decision_vector.txt", "w+") as myfile:
#            for i in range(0,len(s)):
#                myfile.write(str(s[i])+ '\n')
#                        
#                    
#                    
#        ##############
#        # CREATE STL #
#        ##############
#        ctrl_pts_duct = ductPolygon.convert_dec2pos(s[:n_dim])   # the four ends points and the single decision
#        duct_curve_points = ductPolygon.subdc_to_points([ctrl_pts_duct], niterations)
#        print("duct_curve_points", len(duct_curve_points))
#        
#        ctrl_pts_baffle = bafflePolygon.convert_dec2pos(s[n_dim:])
#        baffle_curve_points = bafflePolygon.subdc_to_points([ctrl_pts_baffle], niterations)
#        print("baffle_curve_points", len(baffle_curve_points))
#    
#        ctrl_pts_corner = cornerPolygon.convert_dec2pos_corner(s[n_dim:])
#        corner_curve_points = cornerPolygon.subdc_to_points([ctrl_pts_corner], niterations)
#        print("corner_curve_points", len(corner_curve_points))
#    
#    
#        # write the decision vector to a file        
#        with open(filedirs[-1] + "/duct_curve_points.txt", "w+") as myfile1:
#            for i in range(0,len(duct_curve_points)):
#                myfile1.write(str(duct_curve_points[i][0]) + ',' + str(duct_curve_points[i][1]) + '\n')
#        print("written duct curve points to duct_curve_points.txt")
#    
#    
#        with open(filedirs[-1] + "/baffle_curve_points.txt", "w+") as myfile2:
#            for i in range(0,len(baffle_curve_points)):
#                myfile2.write(str(baffle_curve_points[i][0]) + ',' + str(baffle_curve_points[i][1]) + '\n')
#        print("written baffle curve points to baffle_curve_points.txt")
#    
#        with open(filedirs[-1] + "/corner_curve_points.txt", "w+") as myfile3:
#            for i in range(0,len(corner_curve_points)):
#                myfile3.write(str(corner_curve_points[i][0]) + ',' + str(corner_curve_points[i][1]) + '\n')
#        print("written corner curve points to corner_curve_points.txt")
#        
#        
#        with open(filedirs[-1] + '/headcell_geometry.py', 'r') as f:
#            data = f.readlines()
#        
#        # change the line to the correct directory
#        for line in range(len(data)):       
#            if 'dir_name = None' in data[line]:
#                data[line] = 'dir_name = ' + '"' + dir_name +'"' + '\n'
#        
#        #write the lines to the file        
#        with open(filedirs[-1] + '/headcell_geometry.py', 'w+') as f:
#            f.writelines(data)
#        
#        
#        if(environment == 'local'):
#    
#            subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', '-t', current + case_path + dir_name +'/headcell_geometry.py'])
#            
#            print("salome walls.stl created")   
#            
#            subprocess.call(['/home/andrew/SALOME-9.3.0-UB16.04-SRC/./salome', 'killall'])   
#            
#        elif((environment == 'isca') or (environment == 'isca_test_mesh')):
#    
#            subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', '-t', current + case_path + dir_name +'/headcell_geometry.py'])
#            
#            print("salome walls.stl created") 
#            
#            subprocess.call(['/gpfs/ts0/home/apr207/SALOME-9.3.0-CO7-SRC/./salome', 'killall'])
        
        
            
            
            
            
            
            
            
            
            
            
            
        
        
            
#    fig = plt.figure(figsize=(40,40))
#    SMALL_SIZE = 48
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif') 
#    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
#           
#    ax1=fig.add_subplot(1,1,1)
#    ax1.plot(curve_points[:,0], curve_points[:,1],'rx', lw=1, markersize=15,markeredgewidth=2, label='Point 1')
#    ax1.set_xlim([-45,0])
#    ax1.set_ylim([15,60]) # 883.7 offset because of inlet
#    ax1.legend(loc='upper right', bbox_to_anchor= (1, 1.125), handletextpad=-0.05,handleheight=1.3,ncol=1, borderaxespad=0, frameon=True, fontsize=SMALL_SIZE)
#    plt.tight_layout()
#    ax1.set_xlabel(r'$x \ (mm)$')
#    ax1.set_ylabel(r'$y \ (mm)$')
#    plt.savefig("curve_points.png")
    
     
        ##############
        # CREATE STL #
        ##############

    ######################
    # CREATE DIRECTORIES #
    ######################




    ######################
    # SUBMISSION #
    ######################
    
#    initial_X = [] # decision vector
#    initial_M = [] # checkMesh
#    initial_Y = [] # objectives
#    
#    initial_samples_file = 'initial_samples.npz'
#    
#    subprocess.call(['rm', '-r', current + '/' + initial_samples_file])
#    
#    try:
#        np.savez(current + '/' + initial_samples_file, initial_X, initial_M, initial_Y)
#        print('Data saved in file: ', initial_samples_file)
#    except Exception as e:
#        print(e)
#        print('Data saving failed.')
#    
#    
#    
#    
#    queue(filedirs)
    
    

    ######################
    # SUBMISSION #
    ######################


    #########################
    # BAYESIAN OPTIMISATION #
    #########################


    from IPython.display import clear_output
    import IscaOpt

    # sim_id
    today = str(date.today())
    sim_id = today
    
    # initial samples
    sim_file = 'initial_samples_0.npz'        # put the columns in the right order
    
    # reference vector
#    data = np.load(sim_file)
#    y_objectives = data['arr_1'] # objectives
    
#    cp_base_case = 9.70066
#    gamma_base_case = -0.0153936
    
    
    reference_vector = np.array([0.9, 0.9])
    
    # kernel
    kernel_name = 'Matern52'
    
    # budget
    number_of_LHS_samples = 43 
    number_of_BO_samples = 100  # 2 weeks
    budget = number_of_LHS_samples + number_of_BO_samples
    
    n_dim=4
    
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

    #########################
    # BAYESIAN OPTIMISATION #
    #########################








    ############
    # PLOTTING #
    ############
#    case = Case("./headcell.vtk", pointData=True)                               # 125 micron correction
#    choose_decision_number = 0
#    
#    shape_duct = ductPolygon.convert_dec2pos(lhs_duct[choose_decision_number])
#    points_duct = [shape_duct]
#    for control_points_duct in points_duct:
#        decision_curve_duct = ductPolygon.catmull(control_points_duct)
#        for i in range(niterations):
#            decision_curve_duct = ductPolygon.catmull(decision_curve_duct)
#    
#    shape_baffle = bafflePolygon.convert_dec2pos(lhs_baffle[choose_decision_number])
#    points_baffle = [shape_baffle]
#    for control_points_baffle in points_baffle:
#        decision_curve_baffle = bafflePolygon.catmull(control_points_baffle)
#        for i in range(niterations):
#            decision_curve_baffle = bafflePolygon.catmull(decision_curve_baffle)
#    
#    shape_corner = cornerPolygon.convert_dec2pos_corner(lhs_baffle[choose_decision_number])
#    points_corner = [shape_corner]
#    for control_points_corner in points_corner:
#        decision_curve_corner = cornerPolygon.catmull(control_points_corner)
#        for i in range(niterations):
#            decision_curve_corner = cornerPolygon.catmull(decision_curve_corner)    
#    
#    
#    
#    
#    
#    plot = Plotting(case, corner_boundary, baffle_boundary, duct_boundary, fixed_points, decision_boundary_duct, decision_boundary_baffles, decision_boundary_corner, choose_decision_number)
#    plot.plot_corner_boundary()
#    plot.plot_baffle_boundary()   
#    plot.plot_duct_boundary()
#    
#    
#    
#    plot.plot_decision_and_catmull_clark_curve_baffle(lhs_baffle, decision_curve_baffle)
#    plot.plot_decision_and_catmull_clark_curve_duct(lhs_duct, decision_curve_duct)
#    plot.plot_decision_and_catmull_clark_curve_corner(lhs_baffle, decision_curve_corner)
#
#
#    plot.plot_decision_boundary_and_fixed_points()
#    plot.plot_all_curves(lhs_duct, lhs_baffle)
#    
#    
#    
#    plot.plot_lhs_duct_points()
#    plot.plot_lhs_baffle_points()
    ############
    # PLOTTING #
    ############




    
    print('Total serial wall clock time: ', time.time()-start_sim)
