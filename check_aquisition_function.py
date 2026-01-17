#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:34:00 2022

@author: andrew
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.special import erf as ERF


def plot_contour_2D(func, lb, ub, delta=0.01, nlevels=10, args=(), kwargs={}, \
    ctitle="Function Values", data=None, yrand=None,datac=None, opt=None):
    """
    Plot contour lines of the function outputs. 
    
    Parameters.
    -----------
    func (method): the function that we are interested in
    lb (np.array): lower bounds
    ub (np.array): upper bounds
    delta (float): step size in each dimension
    nlevels (int): number of contour levels.
    args (tuple): arguments to the function.
    kwargs (dictionary): keyword arguments to the function.
    ctitle (str): colourbar title.
    data (m x 2 numpy array): data to scatter on top of the contour plot.
    
    Draws a contour plot.
    """
    plt.figure(figsize=(9,8))
    plt.rc('font', family='serif')
    SMALL_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rcParams['path.simplify'] = False
    
    x = np.arange(lb[0], ub[0]+0.001, delta)
    y = np.arange(lb[1], ub[1]+0.001, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for (i, j), val in np.ndenumerate(Z):
        param = np.array([X[i,j], Y[i,j]]).reshape((1,-1))
        try:
            Z[i,j] = func(param, *args, **kwargs)
        except:
            Z[i,j] = func(param[0], *args, **kwargs)
    vmin = np.min(Z)
    vmax = np.max(Z)

    try:
        levels = np.arange(vmin, vmax + (vmax- vmin) / nlevels, (vmax - vmin)/nlevels)# + ((vmax- vmin) / nlevels) +1e-8, (vmax - vmin)/nlevels)
        norm = mc.BoundaryNorm(levels, 256)
        CS1 = plt.contourf(X, Y, Z, levels=levels, norm=norm, alpha=1, cmap=plt.cm.coolwarm)
    except Exception as e:
        print(e)
        print("Failed to do many levesls.")
        CS1 = plt.contourf(X, Y, Z, alpha=0.75)

    cbar = plt.colorbar(CS1, orientation='vertical', pad=0.05)#, ticks=contour_levels)
    cbar.set_label(ctitle, labelpad=20) 
    cbar.ax.tick_params(which='major', length=10) 
    
    if data is not None:      
        plt.scatter(
        data[:,0], data[:,1],
        s=20**2,                     # markersize**2 to match size
        facecolors='None',           # capital N is safest
        edgecolors=(0, 1, 0),        # explicit lime
        linewidths=3,
        zorder=10
    )
        
    if datac is not None:
        color = [str(item/255.) 
                 for item in 
                 np.linspace(0, 255, len(datac[:,1]))]
        plt.scatter(datac[:,0], datac[:,1],
                   color=color)
    if opt is not None:
        plt.scatter(opt[:,0], opt[:,1],
                   color="orange", marker="*", s=100)
    plt.xlabel('$f_1$ (-)', labelpad=15)
    plt.ylabel('$f_2$ (-)', labelpad=-2) 
    levels=[-1, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    plt.axvline(base[0], ls="dashed", color=(0,1,0), linewidth=3, zorder=9)
    plt.axhline(base[1], ls="dashed", color=(0,1,0), linewidth=3, zorder=9)

    plt.tick_params(axis='both', which='major', length=10)
    plt.xticks(levels)
    plt.yticks(levels)
    plt.plot(yrand[:,0], yrand[:,1], markersize=12, marker="x",mew=2, color="black", ls="None", zorder=100)
    plt.xlim([lb[0], ub[0]])
    plt.ylim([lb[1], ub[1]])
    
    # Example for X axis
    locs, _ = plt.xticks()
    labels = []
    for v in locs:
        if abs(v * 2 - round(v * 2)) < 1e-8 and round(v * 2) % 2 == 1:
            # half values (e.g. 0.5, 1.5, 2.5)
            labels.append(f"{v:.1f}")
        else:
            # whole or near-whole values
            labels.append(f"{v:.1f}")
    plt.xticks(locs, labels)
    
    # Same for Y axis
    locs, _ = plt.yticks()
    labels = []
    for v in locs:
        if abs(v * 2 - round(v * 2)) < 1e-8 and round(v * 2) % 2 == 1:
            labels.append(f"{v:.1f}")
        else:
            labels.append(f"{v:.1f}")
    plt.yticks(locs, labels)
    plt.draw()
    plt.tight_layout()
    
    if(ctitle == "PrefMPoI (-)"):
        plt.savefig("PrefMPoI"+'.png', transparent=False)
    else:
        plt.savefig("MPoI"+'.png', transparent=False)

def displacement(y, r):
    v =  -np.max(np.concatenate([np.max(y-r, axis=1)[:,np.newaxis],np.zeros((y.shape[0], 1))], axis=1), axis=1)
    return v

def get_subsets(y, r, obj_sense=None):
    if obj_sense is None:
        obj_sense = -1*np.ones(y.shape[1])
    tests = np.array([CS.compare_solutions(i, r, obj_sense) for i in y])
    xcl_inds = np.where((tests == 1) | (tests == 3))[0]
    inc_inds = np.where((tests == 0) | (tests == 2))[0]
    return inc_inds, xcl_inds

def mpoi(yp, P, stdp = np.ones((1,2))*0.1):
    '''
    Calculate the minimum probability of improvement compared to current 
    Pareto front. Refer to the paper for full details.

    parameters:
    -----------
    x (np.array): decision vectors.
    cfunc (function): cheap constraint function.
    cargs (tuple): argument for constraint function.
    ckwargs (dict): keyword arguments for constraint function.

    Returns scalarised cost.
    '''
    res = np.zeros((yp.shape[0], 1))
    sqrt2 = np.sqrt(2)
    for i in range(yp.shape[0]):
        m = (yp[i] - P)/(sqrt2 * stdp[i])
        pdom = 1 - np.prod(0.5 * (1 + ERF(m)), axis=1)
        res[i] = np.min(pdom)
    return res


def pref_mpoi(yp, P, base, stdp = np.ones((1,2))*0.1):
    '''
    Calculate the minimum probability of improvement compared to current 
    Pareto front. Refer to the paper for full details.

    parameters:
    -----------
    x (np.array): decision vectors.
    cfunc (function): cheap constraint function.
    cargs (tuple): argument for constraint function.
    ckwargs (dict): keyword arguments for constraint function.

    Returns scalarised cost.
    '''
    res = np.zeros((yp.shape[0], 1))+10**(-11)
    sqrt2 = np.sqrt(2)
    disp = displacement(yp, base)
    
    for i in range(yp.shape[0]):
        m = (yp[i] - P)/(sqrt2 * stdp[i])
        pdom = 1 - np.prod(0.5 * (1 + ERF(m)), axis=1)
        if disp >= 0:
            res[i] = np.min(pdom)
        
    return res

if __name__ == '__main__':

    seed = 12345
    np.random.seed(seed)

    # function settings
    from deap import benchmarks as BM
    fun = BM.dtlz2
    args = (2,) # number of objectives as argument
    
    # random samples
    xrand = np.random.random((20, 6))
    yrand = np.array([fun(i, *args) for i in xrand])
    
    #pareto front
    import _csupport as CS
    nond_ind = CS.nond_ind(yrand, [-1,-1])
    pf_ind = np.where(nond_ind < 1)[0]
    P = yrand[pf_ind]
    
    # base case
    base = np.array([1.1, 1.4])
    inc_ind, xcl_ind = get_subsets(P, base) 
       
    # MPoI plot
    plot_contour_2D(mpoi,[-1,-1], [2.5,2.5], args=(P,), data=P, yrand=yrand, delta=0.1, ctitle="MPoI (-)")
    
    # Reduce Pareto front
    Pareto = P[(P[:, 0] <= base[0]) & (P[:, 1] <= base[1])]   
    
    # PrefMPoI plot
    plot_contour_2D(pref_mpoi,[-1,-1], [2.5,2.5], args=(Pareto,base), data=Pareto, yrand=yrand, delta=0.1, ctitle="PrefMPoI (-)") 
    
       
    