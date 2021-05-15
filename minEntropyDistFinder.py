'''
Created on May 14, 2021

@author: mvelasco
'''
import numpy as np
from optimalTransports import Empirical_Measure,Probability_Measure, Weighted_Voronoi_Diagram
import pdb

class Two_Dim_Aux_Solver:
    def __init__(self,delta,  empirical_measure, probability_measure, distance_fn, num_MC, current_weights):
        #Check the input types:
        assert(isinstance(empirical_measure,Empirical_Measure ))
        self.empirical_measure = empirical_measure
        assert(isinstance(probability_measure,Probability_Measure ))
        self.probability_measure = probability_measure
        centers = empirical_measure.data_vectors
        self.WVD = Weighted_Voronoi_Diagram(centers,current_weights,distance_fn)
        #Initialize the optimization variables
        self.delta = delta
        self.num_MC = num_MC
        self.current_uv = np.array([0.0,1.0])
        self.best_uv = self.current_uv
        assert(self.empirical_measure.ndata_vectors == len(current_weights))
        self.best_objective_so_far = self.compute_objective()
        self.best_gradient_norm =np.linalg.norm(self.compute_gradient())
        
    def compute_objective(self):
        u = self.current_uv[0]
        v = self.current_uv[1]
        value = -u-self.delta*v
        #Create a sample from our probability measure
        sample_vectors_array = self.probability_measure.sample_q(self.num_MC)
        #Evaluate the minimum weighted distance of each sampled point to the centers
        min_distances = self.WVD.compute_array_minimal_weighted_distances(sample_vectors_array)
        values_vector = np.exp((-1)*(v*min_distances + u+1))
        #Monte Carlo evaluation of the integral
        return(value - np.average(values_vector))

    def compute_gradient(self):
        #Create a sample from our probability measure (use only one sample for both components)
        u = self.current_uv[0]
        v = self.current_uv[1]
        sample_vectors_array = self.probability_measure.sample_q(self.num_MC)
        c1 = -1
        c2 = -self.delta
        min_distances = self.WVD.compute_array_minimal_weighted_distances(sample_vectors_array)
        c1_values_vector = np.exp((-1)*(v*min_distances + u+1))
        c2_values_vector = np.multiply(c1_values_vector, min_distances)
        results_array = np.array([c1+np.average(c1_values_vector), c2+np.average(c2_values_vector)])
        return(results_array)
    
    def do_projected_gradient_descent(self,NumSteps, StepSize, keep_track_of_best = True, Descending_in_size=True):
        for k in range(NumSteps):
            gradient = self.compute_gradient()
            if Descending_in_size:    
                CurrStepSize = StepSize/(1+np.sqrt(k))#
            else:
                CurrStepSize = StepSize
            new_uv = self.current_uv + CurrStepSize * gradient #gradient step
            new_uv[1] = max(0.0, new_uv[1]) #proection step
            self.current_uv = new_uv

            if keep_track_of_best:
                objective = self.compute_objective()
                grad_norm = np.linalg.norm(gradient)
                print("Step "+str(k)+":\n")
                self.print_current_status(objective, grad_norm)
                if objective > self.best_objective_so_far:
                    self.best_objective_so_far = objective
                    self.best_uv = new_uv
                    self.best_gradient_norm = np.linalg.norm(gradient)

        #if we keep track of the best then we should set it as the chosen weights
        if keep_track_of_best:
            self.current_uv = self.best_uv
            
    def print_current_status(self,curr_obj,curr_grad_norm):
        print("_____________________________________________________________")
        print("Curr_obj: " + str(curr_obj)+ "")
        print("Curr_grad_norm: " + str(curr_grad_norm)+ "")
        print("Best_obj: " + str(self.best_objective_so_far)+ "")
        print("grad_norm_at_best: " + str(self.best_gradient_norm)+ "")
        print("_____________________________________________________________")
        

"""PROTOTYPES FOR DISTANCE FUNCTION AND SAMPLING FUNCTION FOR SPECIFYING A DISTRIBUTION"""        
def dist(x,y):
    #Prototype for distance functions allowed. For now uses the l2-norm but one can put here any norm
    assert(len(x)==2)
    assert(len(x)==len(y))
    return np.linalg.norm(x-y)    

def sample_q(numSamples):
    #prototype of a sampling function. This is how measures are specified.
    """returns a collection of numSamples many independent vectors unif distributed in [-1,1] x [-1,1]"""
    ResultsArray = []
    Xs = np.random.uniform(-1,1,numSamples)
    Ys = np.random.uniform(-1,1,numSamples)
    for k in range(numSamples):
        ResultsArray.append([Xs[k],Ys[k]])
    return ResultsArray


if __name__ == "__main__":
    #0.Weighted Voronoi diagrams:
    N=5 #number of centers of the empirical measure
    centers_array = [np.random.uniform(-1,1,2) for k in range(N)] #Chosen uniformly in the square [-1,1]x[-1,1]
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(sample_q)
    #We construct the optimal transport object which carries out the gradient descent
    initial_weights = np.zeros(empirical_measure.ndata_vectors)
    num_MC=100000
    delta=0.05
    TDS = Two_Dim_Aux_Solver(delta, empirical_measure,probability_measure,dist,num_MC,initial_weights)
    obj = TDS.compute_objective()
    grad = TDS.compute_gradient()
    print(obj)
    print(grad)
    TDS.do_projected_gradient_descent(NumSteps = 40, StepSize=1.0)
    print(TDS.current_uv)
    print("Done")

