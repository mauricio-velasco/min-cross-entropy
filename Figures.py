'''
Created on May 20, 2021

@author: mvelasco
'''
import pdb
import numpy as np
from optimalTransports import Empirical_Measure, Probability_Measure, Optimal_Transport_Finder, Weighted_Voronoi_Diagram
from optimalTransports import dist, two_d_uniform_density,two_d_uniform_sample_q
from minEntropyDistFinder import norm, Minimum_Cross_Entropy_Dist_Finder

def make_opt_transport_figures():
    #We create two figures, the weighted Voronoi diagram before and after finding the corresponding optimal transport
    # from the uniform distribution in the square to the empirical measure supported at the centers below
    centers_array = [np.array([1/2,5/12]), np.array([7/8,5/12]), np.array([7/8, 6/7]), np.array([7/10, 11/14]),np.array([1/6, 1/2]), np.array([1/4,1/4])]
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(two_d_uniform_sample_q, two_d_uniform_density)  
    OT = Optimal_Transport_Finder(empirical_measure,probability_measure,dist,num_MC=100000)
    OT.WVD.plot_WVD("unweighted_voronoi.png")  
    OT.do_gradient_descent(NumSteps=50, StepSize=0.5, keep_track_of_best=True, Descending_in_size=True)    
    OT.WVD.plot_WVD("optimal_transport.png")

def make_min_entropy_single_figure(delta, filename):
    #The distance between the uniform and the empirical is around 0.2
    centers_array = [np.array([1/2,5/12]), np.array([7/8,5/12]), np.array([7/8, 6/7]), np.array([7/10, 11/14]),np.array([1/6, 1/2]), np.array([1/4,1/4])]
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(two_d_uniform_sample_q, two_d_uniform_density)  
    num_MC=1000    
    #Next we construct the full minimum entropy finder    
    diamK = 1.0
    distance_fn = dist
    dualnorm_fn = norm
    num_MC = 500
    num_UVsteps = 10
    UV_steptype = "backtracking"
    UV_initial_stepsize = 8
    #The following object will carry out the optimization and find the minimum cross entropy distribution for us.
    MCE = Minimum_Cross_Entropy_Dist_Finder(
        delta,  
        diamK, 
        empirical_measure, 
        probability_measure, 
        distance_fn, 
        dualnorm_fn, 
        num_MC,
        num_UVsteps, 
        UV_steptype,
        UV_initial_stepsize        
        )   
    MCE.do_lambda_star_several_steps(numsteps=50, verbose=True)
    #Results (which encode the minimum cross entropy distribution via q = p*exp(-1-v\phi_{\lambda}(x)-u)
    #where p is the density of the probability distribution above
    print("Final_uv: " + str(MCE.best_uv))
    print("Best_lambdas: "+str(MCE.current_lambdas))
    print("Gradient_norm : "+str(MCE.TDS.best_gradient_norm) )
    print("Gradient : "+str(MCE.TDS.gradient) )
    print("Cross-Entropy : "+str(MCE.TDS.best_objective_so_far) )
    
    MCE.plot_current_weights_dist(filename, num_points=300)

def make_min_entropy_figures():
    delta_values = [0.2, 0.15, 0.1, 0.05, 0.02, 0.0005]
    for k, delta in enumerate(delta_values):
        make_min_entropy_single_figure(delta, "min_entropy_"+str(k)+".png")
        
        
#One selects which figures to create below
#make_opt_transport_figures()
#make_min_entropy_single_figure(delta = 0.2,filename = "min_entropy_0.png")
make_min_entropy_figures()
