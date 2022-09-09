#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:53:20 2022

@author: jitendra
"""



import numpy as np
import mdptoolbox, mdptoolbox.example
import random
from numpy.random import seed
import time
from datetime import datetime
import sys



s =50
a =25

discount = 0.9
episodes = 20

iterations=5000

SET = 10**(-6) #successive error threshold
w_val = 1.3

vi_diff = np.zeros((episodes,iterations,1))
newton_diff= np.zeros((episodes,iterations,1))
gen_newton_diff = np.zeros((episodes,iterations,1))

# vi_pol_diff = np.zeros((episodes,iterations,1))
# newton_pol_diff = np.zeros((episodes,iterations,1))

vi_time = np.zeros((episodes,1))
newton_time = np.zeros((episodes,1))
gen_newton_time = np.zeros((episodes,1))
Newton_JacobiQValueIteration_time=np.zeros((episodes,1))
Newton_GSQValueIteration_time=np.zeros((episodes,1))
ModifiedNewton_time=np.zeros((episodes,1))
GSModifiedNewtonQValueIteration_time=np.zeros((episodes,1))
JacobiModifiedNewtonQValueIteration_time=np.zeros((episodes,1))
vi_error = np.zeros((episodes,1))
newton_error = np.zeros((episodes,1))
gen_newton_error = np.zeros((episodes,1))
Newton_JacobiQValueIteration_error=np.zeros((episodes,1))
Newton_GSQValueIteration_error=np.zeros((episodes,1))
ModifiedNewton_error=np.zeros((episodes,1))
JacobiModifiedNewtonQValueIteration_error=np.zeros((episodes,1))
GSModifiedNewtonQValueIteration_error=np.zeros((episodes,1))
smooth_picard_time=np.zeros((episodes,1))
smooth_picard_error = np.zeros((episodes,1))
newtonsmooth_picard_time=np.zeros((episodes,1))
newtonsmooth_picard_error = np.zeros((episodes,1))
gn=np.zeros((episodes,1))
gs=np.zeros((episodes,1))
nj=np.zeros((episodes,1))

mn=np.zeros((episodes,1))
sp=np.zeros((episodes,1))
jsp=np.zeros((episodes,1))
jmn=np.zeros((episodes,1))
gsmn=np.zeros((episodes,1))

print("S =",s, "\nA =",a,"\nEpisodes =",episodes, "\nMax Iterations =",iterations,"\nSET =",SET,"\nGamma =",discount)

for count in range(episodes):

    print("*********** Episode",count+1,"************")
    np.random.seed((count + 1)*100)
    random.seed((count + 1)*110)
   
    P, R = mdptoolbox.example.rand(s, a)

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.0000001,max_iter = iterations)
    vi.run()
    # print(vi.V)
    # print(vi.V)
    #start = time.time()
    #vi3 = mdptoolbox.mdp.QValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy)
    
    
    #vi3.run()
    #end = time.time()
    #vi_time[count] = end-start

    #start = time.time()
    #vi4 = mdptoolbox.mdp.NewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy)
    #vi4.run()
    #end = time.time()
    
    
    
    #newton_time[count] = end-start
    
    start = time.time()
    vi5 = mdptoolbox.mdp.GenNewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi5.run()
    end = time.time()
    gen_newton_time[count] = end-start
    #print("Gen_Newtontime",gen_newton_time)
    
    
    start=time.time()
    vi7 = mdptoolbox.mdp.NewtonGSQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi7.run()
    end = time.time()
    Newton_GSQValueIteration_time[count] = end-start
    
    
    start=time.time()
    vi6 = mdptoolbox.mdp.NewtonJacobiQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    
    vi6.run()
    end = time.time()
    Newton_JacobiQValueIteration_time[count] = end-start
    #print('NJacobitime' ,Newton_JacobiQValueIteration_time)
    
    
    
    #print('GaussSeideltime' ,Newton_GSQValueIteration_time)
    
    start=time.time()
    vi8 = mdptoolbox.mdp.ModifiedNewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi8.run()
    end = time.time()
    ModifiedNewton_time[count] = end-start
    
    
    start = time.time()
    vi9 = mdptoolbox.mdp.SmoothPicardQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi9.run()
    end = time.time()
    smooth_picard_time[count]=end-start
    
    """
    
    start = time.time()
    vi10 = mdptoolbox.mdp.JacobiModifiedNewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi10.run()
    end = time.time()
    JacobiModifiedNewtonQValueIteration_time[count]=end-start
    
    start = time.time()
    vi11 = mdptoolbox.mdp.GSModifiedNewtonQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi11.run()
    end = time.time()
    GSModifiedNewtonQValueIteration_time[count]=end-start
    """
    start = time.time()
    vi12 = mdptoolbox.mdp.NewtonSmoothPicardQValueIteration(P,R,discount,max_iter = iterations,opt = vi.V, opt_pol = vi.policy,w = w_val,successive_error_threshold = SET)
    vi12.run()
    end = time.time()
    newtonsmooth_picard_time[count]=end-start
    # print('GenNewtonQValueIteration')
    gen_newton_error[count]=np.linalg.norm(vi.V - np.max(vi5.nmodQ,axis = 0),axis=0,ord=np.inf)
    #print(np.linalg.norm(vi.V - np.max(vi5.nmodQ,axis = 0),axis=0,ord=np.inf))
    # print(vi5.iter)
    gn[count]=vi5.iter
    

    # print('NewtonGSQValueIteration')
    #print(np.linalg.norm(vi.V - np.max(vi7.nmodQ,axis = 0),axis=0,ord=np.inf))
    # print(vi7.iter)
    Newton_GSQValueIteration_error[count]=np.linalg.norm(vi.V - np.max(vi7.nmodQ,axis = 0),axis=0,ord=np.inf)
    gs[count]=vi7.iter
    # print('NewtonJacobiQValueIteration')
    #print(np.linalg.norm(vi.V - np.max(vi6.nmodQ,axis = 0),axis=0,ord=np.inf))
    Newton_JacobiQValueIteration_error[count]=np.linalg.norm(vi.V - np.max(vi6.nmodQ,axis = 0),axis=0,ord=np.inf)
    # print(vi6.iter)
    nj[count]=vi6.iter
    
    
    # print('ModifiedNewtonQValueIteration')
    ModifiedNewton_error[count]=np.linalg.norm(vi.V - np.max(vi8.nmodQ,axis = 0),axis=0,ord=np.inf)
    #print(np.linalg.norm(vi.V - np.max(vi8.nmodQ,axis = 0),axis=0,ord=np.inf))
    mn[count]=vi8.iter
    # print(vi8.iter)   
    
    
    # print('smooth picard')
    smooth_picard_error[count]=np.linalg.norm(vi.V - np.max(vi9.nmodQ,axis = 0),axis=0,ord=np.inf)
    # print(vi9.iter)
    sp[count]=vi9.iter
    
    newtonsmooth_picard_error[count]=np.linalg.norm(vi.V - np.max(vi12.nmodQ,axis = 0),axis=0,ord=np.inf)
    jsp[count]=vi12.iter
    """
    # print('Jacobi ModifiedNewtonQValueIteration')
    JacobiModifiedNewtonQValueIteration_error[count]=np.linalg.norm(vi.V - np.max(vi10.nmodQ,axis = 0),axis=0,ord=np.inf)
    # print(vi10.iter)
    jmn[count]=vi10.iter
    
    # print('GS ModifiedNewtonQValueIteration')
    GSModifiedNewtonQValueIteration_error[count]=np.linalg.norm(vi.V - np.max(vi11.nmodQ,axis = 0),axis=0,ord=np.inf)
    # print(vi11.iter)
    gsmn[count]=vi11.iter
    """
#print(np.linalg.norm(vi.V - np.max(vi3.Q,axis = 0),ord=np.inf)) #Q-bellman
#print(np.linalg.norm(vi.V - np.max(vi4.nmodQ,axis = 0),axis=0,ord=np.inf))
print("---------------------")
sum5=0   
error5=0
for i in range(episodes):
        sum5=sum5+gen_newton_time[i]
        error5=error5+gen_newton_error[i]
print('Genaralised Newton average time is', sum5/episodes)  
print('Genaralised Newton average error is', error5/episodes)  
print(np.min(gn))
print(np.max(gn))
print("---------------------")
sum7=0 
error7=0   
for i in range(episodes):
        sum7=sum7+Newton_GSQValueIteration_time[i]
        error7=error7+Newton_GSQValueIteration_error[i]
print('Gauss Seidel average time is',sum7/episodes) 
print('Gauss Seidel average error is',error7/episodes) 
print(np.min(gs))
print(np.max(gs))
print("---------------------")
sum6=0  
error6=0  
for i in range(episodes):
        sum6=sum6+Newton_JacobiQValueIteration_time[i]
        error6=error6+Newton_JacobiQValueIteration_error[i]
print('Newton Jacobi average time is',sum6/episodes) 
print('Newton Jacobi average error is',error6/episodes) 
print(np.min(nj))
print(np.max(nj))
print("---------------------")
sum8=0 
error8=0   
for i in range(episodes):
        sum8=sum8+ModifiedNewton_time[i]
        error8=error8+ModifiedNewton_error[i]
print('Modified Newton average time',sum8/episodes)            
 
print('Modified Newton average error',error8/episodes)   
print(np.min(mn))
print(np.max(mn)) 

print("---------------------")
sum9=0 
error9=0   
for i in range(episodes):
        sum9=sum9+smooth_picard_time[i]
        error9=error9+smooth_picard_error[i]
print('Smooth Picard average time',sum9/episodes)            
 
print('Smooth Picard average error',error9/episodes)  
print(np.min(sp))
print(np.max(sp))
"""
print("---------------------")
sum10=0 
error10=0   
for i in range(episodes):
        sum10=sum10+JacobiModifiedNewtonQValueIteration_time[i]
        error10=error10+JacobiModifiedNewtonQValueIteration_error[i]
print('JacobiModifiedNewton average time',sum10/episodes)            
 
print('JacobiModifiedNewton average error',error10/episodes)  
print(np.min(jmn))
print(np.max(jmn))
print("---------------------")
sum11=0 
error11=0   
for i in range(episodes):
        sum11=sum11+GSModifiedNewtonQValueIteration_time[i]
        error11=error11+GSModifiedNewtonQValueIteration_error[i]
print('GSModifiedNewton average time',sum11/episodes)            
 
print('GSModifiedNewton average error',error11/episodes)  
print(np.min(gsmn))
print(np.max(gsmn))
print("---------------------")
"""
sum12=0 
error12=0   
for i in range(episodes):
        sum12=sum12+newtonsmooth_picard_time[i]
        error12=error12+newtonsmooth_picard_error[i]
print('Newton Picard average time',sum12/episodes)            
 
print('Newton Picard average error',error12/episodes)  
print(np.min(jsp))
print(np.max(jsp))
