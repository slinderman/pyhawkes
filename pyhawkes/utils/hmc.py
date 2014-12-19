"""
Implementation of Hybrid Monte Carlo (HMC) sampling algorithm following Neal (2010).
Use the log probability and the gradient of the log prob to navigate the distribution.
"""
import numpy as np
import logging
log = logging.getLogger("global_log")

def hmc(U, grad_U, epsilon, L, q_curr):
    """
    U       - function handle to compute log probability we are sampling
    grad_U  - function handle to compute the gradient of the density with respect 
              to relevant params
    epsilon - step size
    L       - number of steps to take
    q_curr  - current state
    
    """
    # Start at current state
    q = q_curr
    # Moment is simplest for a normal rv
    p = np.random.randn(np.size(q))
    p_curr = p
    
    # Evaluate potential and kinetic energies at start of trajectory
    U_curr = U(q_curr)
    K_curr = np.sum(p_curr**2)/2
    
    # Make a half step in the momentum variable
    p -= epsilon*grad_U(q)/2
    
    # Alternate L full steps for position and momentum
    for i in np.arange(L):
        q += epsilon*p
        
        # Full step for momentum except for last iteration
        if i < L-1:
            p -= epsilon*grad_U(q)
        else:
            p -= epsilon*grad_U(q)/2
    
    # Negate the momentum at the end of the trajectory to make proposal symmetric?
    p = -p
    
    # Evaluate potential and kinetic energies at end of trajectory
    U_prop = U(q)
    K_prop = np.sum(p**2)/2
    
    # Accept or reject new state with probability proportional to change in energy.
    # Ideally this will be nearly 0, but forward Euler integration introduced errors.
    # Exponentiate a value near zero and get nearly 100% chance of acceptance.
    if np.log(np.random.rand()) < U_curr-U_prop + K_curr-K_prop:
        q_next = q
    else:
        q_next = q_curr
        
    return np.reshape(q_next, np.shape(q))