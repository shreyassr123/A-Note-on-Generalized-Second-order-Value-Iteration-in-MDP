#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment script for comparing Value Iteration and Newton-based Q-value 
iteration methods on random MDPs.

Created on Thu Aug 11 16:53:20 2022
@author: Shreyas
"""

import numpy as np
import random
import time
import mdptoolbox, mdptoolbox.example


# ----------------------------
# Experiment Configuration (Can be varied)
# ----------------------------
S, A       = 50, 25             # States and actions 
discount   = 0.9                 # Discount factor
episodes   = 20                  # Number of experiments
iterations = 5000                # Max iterations
SET        = 1e-6                # Successive error threshold
w_val      = 1.3                  # Relaxation parameter


# ----------------------------
# Storage for results
# ----------------------------
Tot_time1, Tot_error1, Tot_iter1 = [], [], []
Tot_time2, Tot_error2, Tot_iter2 = [], [], []
Tot_time3, Tot_error3, Tot_iter3 = [], [], []
Tot_time4, Tot_error4, Tot_iter4 = [], [], []
Tot_time5, Tot_error5, Tot_iter5 = [], [], []
Tot_time6, Tot_error6, Tot_iter6 = [], [], []


# ----------------------------
# Main Experiment Loop
# ----------------------------
print("S =", S, "A =", A, "episodes =", episodes, "iterations =", iterations,
      "SET =", SET, "Gamma =", discount, "\n")

for ep in range(1, episodes + 1):

    print("***********", ep, "************")

    np.random.seed(ep * 100)
    random.seed(ep * 110)

    P, R = mdptoolbox.example.rand(S, A)

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,
                                       epsilon=1e-7,
                                       max_iter=iterations)
    vi.run()

    # ----------------------------
    # GenNewton Q-value iteration
    # ----------------------------
    start = time.time()
    GenNewtonQ = mdptoolbox.mdp.GenNewtonQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    GenNewtonQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(GenNewtonQ.nmodQ, axis=0), ord=np.inf)

    Tot_time1.append(t)
    Tot_error1.append(error)
    Tot_iter1.append(GenNewtonQ.iter)

    # ----------------------------
    # NewtonGS Q-value iteration
    # ----------------------------
    start = time.time()
    NewtonGSQ = mdptoolbox.mdp.NewtonGSQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    NewtonGSQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(NewtonGSQ.nmodQ, axis=0), ord=np.inf)

    Tot_time2.append(t)
    Tot_error2.append(error)
    Tot_iter2.append(NewtonGSQ.iter)

    # ----------------------------
    # NewtonJacobi Q-value iteration
    # ----------------------------
    start = time.time()
    NewtonJacobiQ = mdptoolbox.mdp.NewtonJacobiQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    NewtonJacobiQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(NewtonJacobiQ.nmodQ, axis=0), ord=np.inf)

    Tot_time3.append(t)
    Tot_error3.append(error)
    Tot_iter3.append(NewtonJacobiQ.iter)

    # ----------------------------
    # ModifiedNewton Q-value iteration
    # ----------------------------
    start = time.time()
    ModifiedNewtonQ = mdptoolbox.mdp.ModifiedNewtonQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    ModifiedNewtonQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(ModifiedNewtonQ.nmodQ, axis=0), ord=np.inf)

    Tot_time4.append(t)
    Tot_error4.append(error)
    Tot_iter4.append(ModifiedNewtonQ.iter)

    # ----------------------------
    # SmoothPicard Q-value iteration
    # ----------------------------
    start = time.time()
    SmoothPicardQ = mdptoolbox.mdp.SmoothPicardQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    SmoothPicardQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(SmoothPicardQ.nmodQ, axis=0), ord=np.inf)

    Tot_time5.append(t)
    Tot_error5.append(error)
    Tot_iter5.append(SmoothPicardQ.iter)

    # ----------------------------
    # NewtonSmoothPicard Q-value iteration
    # ----------------------------
    start = time.time()
    NewtonSmoothPicardQ = mdptoolbox.mdp.NewtonSmoothPicardQValueIteration(
        P, R, discount, max_iter=iterations, opt=vi.V,
        opt_pol=vi.policy, w=w_val,
        successive_error_threshold=SET
    )
    NewtonSmoothPicardQ.run()
    t = time.time() - start

    error = np.linalg.norm(vi.V - np.max(NewtonSmoothPicardQ.nmodQ, axis=0), ord=np.inf)

    Tot_time6.append(t)
    Tot_error6.append(error)
    Tot_iter6.append(NewtonSmoothPicardQ.iter)


# ----------------------------
# Final Summary
# ----------------------------
print("\n\n======================= AVERAGE RESULTS =======================\n")

print("GenNewtonQ Results:")
print("Time =", np.mean(Tot_time1))
print("Error =", np.mean(Tot_error1))
print("Iterations min =", np.min(Tot_iter1), "max =", np.max(Tot_iter1))

print("\nNewtonGSQ Results:")
print("Time =", np.mean(Tot_time2))
print("Error =", np.mean(Tot_error2))
print("Iterations min =", np.min(Tot_iter2), "max =", np.max(Tot_iter2))

print("\nNewtonJacobiQ Results:")
print("Time =", np.mean(Tot_time3))
print("Error =", np.mean(Tot_error3))
print("Iterations min =", np.min(Tot_iter3), "max =", np.max(Tot_iter3))

print("\nModifiedNewtonQ Results:")
print("Time =", np.mean(Tot_time4))
print("Error =", np.mean(Tot_error4))
print("Iterations min =", np.min(Tot_iter4), "max =", np.max(Tot_iter4))

print("\nSmoothPicardQ Results:")
print("Time =", np.mean(Tot_time5))
print("Error =", np.mean(Tot_error5))
print("Iterations min =", np.min(Tot_iter5), "max =", np.max(Tot_iter5))

print("\nNewtonSmoothPicardQ Results:")
print("Time =", np.mean(Tot_time6))
print("Error =", np.mean(Tot_error6))
print("Iterations min =", np.min(Tot_iter6), "max =", np.max(Tot_iter6))
