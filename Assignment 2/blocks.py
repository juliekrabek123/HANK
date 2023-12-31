import numpy as np
import numba as nb

from GEModelTools import prev,next

import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production_firm(par,ini,ss,K,L_Y,rK,w,Y):

    K_lag = lag(ini.K,K)

    # a. implied prices (remember K and L are inputs)
    rK[:] = par.alpha*par.Gamma_Y*(K_lag/L_Y)**(par.alpha-1.0)

    w[:] = (1.0-par.alpha)*par.Gamma_Y*(K_lag/L_Y)**par.alpha
    
    # b. production and investment
    Y[:] = par.Gamma_Y*K_lag**(par.alpha)*L_Y**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rK,A,r):

    # a. total assets
    A[:] = K + ss.B

    # b. return
    r[:] = rK-par.delta

@nb.njit
def government(par,ini,ss,B,tau,w,wt,S,G,L_G,Gamma_G,chi):

    tau[:] = ss.tau
    B[:] = ss.B
    wt[:] = (1-tau)*w
    S[:] = np.min([G[:], Gamma_G*L_G])

    # b. Transfer
    chi[:] = ss.chi


@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L_G,L_Y,L_hh,Y,C_hh,K,I,G,tau,w,chi,clearing_A,clearing_L,clearing_Y,clearing_G):

    clearing_A[:] = A - A_hh
    #L[:] = L_hh
    clearing_L[:] = L_hh - (L_Y + L_G) 
    I[:] = K - (1-par.delta) * lag(ini.K,K)
    clearing_Y[:] = Y - C_hh - I - G
    clearing_G[:] = (G + w*L_G + chi) - (tau * w * L_hh)