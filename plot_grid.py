import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#credits: https://medium.com/analytics-vidhya/monte-carlo-methods-in-reinforcement-learning-part-2-off-policy-methods-b9766d3e60d7

def plot_state_value_grid(state_value_grid):
    """ Plots the State_Value_Grid """
    plt.figure(figsize=(10,5))
    p=sns.heatmap(state_value_grid, cmap='coolwarm', annot=True, fmt=".1f",annot_kws={'size':16},square=True)
    p.set_ylim(len(state_value_grid)+0.01, -0.01) # because of seaborn problem 
    
def plot_action_value(action_value_grid):
    top=action_value_grid[:,0].reshape((3,4))
    top_value_positions = [(0.38,0.25),(1.38,0.25),(2.38,0.25),(3.38,0.25),
                           (0.38,1.25),(1.38,1.25),(2.38,1.25),(3.38,1.25),
                           (0.38,2.25),(1.38,2.25),(2.38,2.25),(3.38,2.25)]
    right=action_value_grid[:,1].reshape((3,4))
    right_value_positions = [(0.65,0.5),(1.65,0.5),(2.65,0.5),(3.65,0.5),
                           (0.65,1.5),(1.65,1.5),(2.65,1.5),(3.65,1.5),
                           (0.65,2.5),(1.65,2.5),(2.65,2.5),(3.65,2.5)]
    bottom=action_value_grid[:,2].reshape((3,4))
    bottom_value_positions = [(0.38,0.8),(1.38,0.8),(2.38,0.8),(3.38,0.8),
                           (0.38,1.8),(1.38,1.8),(2.38,1.8),(3.38,1.8),
                           (0.38,2.8),(1.38,2.8),(2.38,2.8),(3.38,2.8)]
    left=action_value_grid[:,3].reshape((3,4))
    left_value_positions = [(0.05,0.5),(1.05,0.5),(2.05,0.5),(3.05,0.5),
                           (0.05,1.5),(1.05,1.5),(2.05,1.5),(3.05,1.5),
                           (0.05,2.5),(1.05,2.5),(2.05,2.5),(3.05,2.5)]


    fig, ax=plt.subplots(figsize=(12,5))
    ax.set_ylim(3, 0)
    tripcolor = quatromatrix(left, top, right, bottom, ax=ax,
                 triplotkw={"color":"k", "lw":1},
                 tripcolorkw={"cmap": "coolwarm"}) 

    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    for i, (xi,yi) in enumerate(top_value_positions):
        plt.text(xi,yi,round(top.flatten()[i],2), size=11, color="w")
    for i, (xi,yi) in enumerate(right_value_positions):
        plt.text(xi,yi,round(right.flatten()[i],2), size=11, color="w")
    for i, (xi,yi) in enumerate(left_value_positions):
        plt.text(xi,yi,round(left.flatten()[i],2), size=11, color="w")
    for i, (xi,yi) in enumerate(bottom_value_positions):
        plt.text(xi,yi,round(bottom.flatten()[i],2), size=11, color="w")

    plt.show()
    
    
def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):

    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor

def action_probs(A):
    epsilon = 0.25
    idx = np.argmax(A)
    probs = []
    A_ = np.sqrt(sum([i**2 for i in A]))

    if A_ == 0:
        A_ = 1.0
    for i, a in enumerate(A):
        if i == idx:
            probs.append(round(1-epsilon + (epsilon/A_),3))
        else:
            probs.append(round(epsilon/A_,3))
    err = sum(probs)-1
    substracter = err/len(A)

    return np.array(probs)-substracter
