'''
Created on May 14, 2021

@author: mvelasco
'''
import numpy as np
import pdb
import time
import numpy as np
import matplotlib.pyplot as plt


class Empirical_Measure:
    """
    An empirical measure is determined by a list of data points (self.data_vectors)
    in some R^n. n=self.dim. Represents the sum of Dirac deltas at these data points.
    
    An empirical measure is capable of:
        -Acting on functions by integration (integrate)
        -Find the index of nearest data point to a given new point (nearest_data_point_index).
            the notion of distance used is specified in self.distance.
        -Classify_nearest (for each vector on a list returns the vector in self.data_vectors closest to it)
    
    """
    def __init__(self, data_points_vector):
        self.data_vectors = []
        self.ndata_vectors = len(data_points_vector)
        assert(self.ndata_vectors>0)
        #Receives an array of vectors corresponding to the observations x_i
        first =True
        for vector in data_points_vector:
            if first:
                first=False
                self.dimension = len(vector)
            assert(len(vector) == self.dimension)
            self.data_vectors.append(np.array(vector))
        self.dim = self.dimension

    def integrate(self,fn):
        Result = []
        for vector in self.data_vectors:            
            Result.append(fn(vector)*(1/self.ndata_vectors))
        return np.sum(Result)


class Probability_Measure:
    """
    We represent probability measures via black box routine that produces independent samples from 
    this distribution and an analytic formula for its density wrt Lebesgue.
    
    
    A probability measure is capable of:
        -Producing an independent sample of the random variable.
        -Acting on functions by integration (integrate via MonteCarlo)
        -evaluate the corresponding probability density function
    
    """
    
    def __init__(self, sample_q_fn, density_fn):
        self.sample_q_fn = sample_q_fn
        self.density_fn = density_fn
        
    def sample_q(self, numsamples):
        return self.sample_q_fn(numsamples)

    def integrate(self, fn):
        points = self.sample_q(self.MC_points)
        return np.average([fn(p) for p in points])
    
    def evaluate_density_fn(self,vector):
        return(self.density_fn(vector))


class Weighted_Voronoi_Diagram:
    """A weighted Voronoi diagram is specified by:
         -a collection of centers 
         -with a corresponding vector of weights, one per center which sum to zero (\lambdas in the article)
         -An ambient distance function

    It should be capable of:
    (1) Given a new vector find the index of the weighted nearest center+
    (2) Given a collection of vectors count the fraction of these points in each of the weighted voronoi cells.+
    (3) Given a collection of vectors and a (u,v) distortion, count the fraction of these points 
    in each of the weighted voronoi cells.+
    """
    
    def __init__(self, centers_array, weights_vector, distance_fn):
        self.centers = []
        self.distance = distance_fn
        self.ncenters = len(centers_array)
        #Receives an array of vectors corresponding to the centers of the Weighted Voronoi
        first =True
        for vector in centers_array:
            if first:
                first=False
                self.dim = len(vector)
            assert(len(vector) == self.dim)
            self.centers.append(np.array(vector))
        self.weights = weights_vector
        assert(len(self.centers) == len(self.weights))

        
    def index_weighted_cell_of(self, vector):
        #Given a new vector find the index of the weighted nearest center        
        N = self.ncenters
        weighted_distances = np.array([self.distance(vector, self.centers[k])-self.weights[k] for k in range(N)])
        index = np.where(weighted_distances == np.min(weighted_distances))[0]
        return(index[0])

    def minimal_weighted_distance_to_data_point(self,vector):
            N = self.ncenters
            weighted_distances = np.array([self.distance(vector, self.centers[k])-self.weights[k] for k in range(N)])
            return(np.min(weighted_distances))


    def compute_array_minimal_weighted_distances(self, vectors_array):
        distances_vector = np.zeros(len(vectors_array))
        for k, vector in enumerate(vectors_array):
            distances_vector[k] = self.minimal_weighted_distance_to_data_point(vector)
        return(distances_vector)

    def compute_array_index_weighted_cell_of(self,vectors_array):
        indices_vector = np.zeros(len(vectors_array))
        for k, vector in enumerate(vectors_array):
            indices_vector[k] = self.index_weighted_cell_of(vector)
        return(indices_vector)

    def compute_proportions_in_cells(self, vectors_array, gradient=False):
        #Given a collection of vectors count the fraction of these points in each of the weighted voronoi cells
        region_counts_vector = np.zeros(self.ncenters)
        for vector in vectors_array:
            idx = self.index_weighted_cell_of(vector)
            region_counts_vector[idx]+=1
        if gradient:
            return (-1)*(region_counts_vector/len(vectors_array)-(1/self.ncenters))
        else:
            return region_counts_vector/len(vectors_array)


    #The distorted weights below are needed for computing Theorem 1.3 b
    def uv_distorted_weight(self,vector, UVvector):
            N = self.ncenters
            weighted_distances = np.array([self.distance(vector, self.centers[k])-self.weights[k] for k in range(N)])
            phi_lambda = np.min(weighted_distances)
            u=UVvector[0]
            v=UVvector[1]
            return(np.exp(-1-v*phi_lambda-u))

    def compute_array_uv_distorted_weights(self, vectors_array, UVvector):
        distorsions_vector = np.zeros(len(vectors_array))
        for k, vector in enumerate(vectors_array):
            distorsions_vector[k] = self.uv_distorted_weight(vector, UVvector)
        return(distorsions_vector)

    def compute_gradient_from_distorted_averages_in_cells(self, vectors_array, UVvector):
        """Given a collection of vectors and a u,v pair compute the average of the uv-distorted distribution in each weighted Voronoi cell"""        
        N_samples=len(vectors_array)
        distortions_vector = self.compute_array_uv_distorted_weights(vectors_array, UVvector)#computes the value of exp(-1-v\phi(z)-u) for z in sample
        indices_vector = self.compute_array_index_weighted_cell_of(vectors_array)# Finds the nearest center to each sample point        
        #Next we go compute the average of the values in each region. Results are stored in region_values_vector
        region_values_vector = np.zeros(self.ncenters)
        for k in range(self.ncenters):
            active_indices = np.where(indices_vector==k)
            if active_indices[0].size > 0:
                region_values_vector[k] = np.sum(distortions_vector[active_indices])/N_samples
        #Finally we return the numerically computed gradient.
        return ((-1)*region_values_vector +1/self.ncenters)#Compute the gradient vector

    def plot_WVD(self, namestring, num_points = 500):        
        plt.figure(dpi=150)
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(1.0)
        M = num_points
        #We will use an MxM grid of points
        xvalues = np.linspace(0,1,num=M);
        yvalues = np.linspace(0,1,num=M);        
        xx, yy = np.meshgrid(xvalues, yvalues)
        g = lambda a,b: self.index_weighted_cell_of(np.array([a,b]))
        Vg = np.vectorize(g)#Vectorization to be able to apply it to a numpy meshgrid
        z = Vg(xx,yy)
        plt.pcolormesh(xvalues, yvalues, z, cmap="RdYlBu_r")
        #Next we plot the centers in black
        Xs = [self.centers[k][0] for k in range(self.ncenters)]
        Ys = [self.centers[k][1] for k in range(self.ncenters)]
        plt.scatter(Xs, Ys, c="black", alpha=0.9) 
        #Finally we save the image and show it to screen
        plt.savefig(namestring)
        plt.show()

class Optimal_Transport_Finder:
    """ This function finds an optimal transport between 
    an empirical measure and any probability measure q specified by a sampling function."""    
    def __init__(self, empirical_measure, probability_measure, distance_fn, num_MC):
        #Check the input types:
        assert(isinstance(empirical_measure,Empirical_Measure ))
        self.empirical_measure = empirical_measure
        assert(isinstance(probability_measure,Probability_Measure ))
        self.probability_measure = probability_measure
        weights = np.zeros(empirical_measure.ndata_vectors)
        centers = empirical_measure.data_vectors
        self.WVD = Weighted_Voronoi_Diagram(centers,weights,distance_fn)
        self.bestLambdas = self.WVD.weights.copy()#Will keeptrack of the best lambdas so far.
        self.num_MC = num_MC
        self.best_objective_so_far = self.compute_objective()
        self.best_weights_so_far = np.zeros(len(self.WVD.weights))
        self.best_gradient_norm = 0
        
    def compute_objective(self):
        """Computes the current \Psi(\lambda) with \lambda=self.WVD.weights"""
        #Create a sample from our probability measure
        sample_vectors_array = self.probability_measure.sample_q(self.num_MC)
        #Evaluate the minimum weighted distance of each sampled point to the centers
        min_distances = self.WVD.compute_array_minimal_weighted_distances(sample_vectors_array)
        #Monte Carlo evaluation of the integral
        return(np.average(min_distances))
 
    def compute_gradient(self):
        #Create a sample from our probability measure
        sample_vectors_array = self.probability_measure.sample_q(self.num_MC)
        gradient = self.WVD.compute_proportions_in_cells(sample_vectors_array, gradient=True)
        return gradient

    def do_gradient_descent(self, NumSteps, StepSize, keep_track_of_best = True, Descending_in_size=True):
        #Do a gradient descent for NumSteps steps
        for k in range(NumSteps):
            weights = self.WVD.weights
            gradient = self.compute_gradient()
            if Descending_in_size:    
                CurrStepSize = StepSize/(1+np.sqrt(k))#
            else:
                CurrStepSize = StepSize
            #gradient step
            new_weights = weights + CurrStepSize * gradient
            self.WVD.weights = new_weights
            if keep_track_of_best:
                objective = self.compute_objective()
                grad_norm = np.linalg.norm(gradient)
                print("Optimal transport descent computation step "+str(k)+":\n")
                self.print_current_status(objective, grad_norm)
                if objective > self.best_objective_so_far:
                #if grad_norm < self.best_gradient_norm:# we are looking for points with Voronoi regions of constant uniform area
                    self.best_objective_so_far = objective
                    self.best_weights_so_far = new_weights
                    self.best_gradient_norm = np.linalg.norm(gradient)
                
        
        #if we keep track of the best then we should set it as the chosen weights
        if keep_track_of_best:
            self.WVD.weights = self.best_weights_so_far

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

def two_d_uniform_sample_q(numSamples):
    #prototype of a sampling function. This is how measures are specified.
    """returns a collection of numSamples many independent vectors unif distributed in [-1,1] x [-1,1]"""
    ResultsArray = []
    Xs = np.random.uniform(0,1,numSamples)
    Ys = np.random.uniform(0,1,numSamples)
    for k in range(numSamples):
        ResultsArray.append([Xs[k],Ys[k]])
    return ResultsArray

def two_d_uniform_density(vector):
    #prototype of a density function. This is how measures are specified.
    """returns the density of an independent 2d vector unif distributed in [-1,1] x [-1,1]"""
    x = vector[0]
    y = vector[1]
    if (0<=x) and (x<=1) and (0<=y) and (y<=1):
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    #0.Weighted Voronoi diagrams:
    N=5 #number of centers of the empirical measure
    centers_array = [np.random.uniform(-1,1,2) for k in range(N)] #Chosen uniformly in the square [-1,1]x[-1,1]
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(two_d_uniform_sample_q,two_d_uniform_density)
    #We construct the optimal transport object which carries out the gradient descent
    OT = Optimal_Transport_Finder(empirical_measure,probability_measure,dist,num_MC=100000)
    OT.do_gradient_descent(NumSteps=30, StepSize=0.5, keep_track_of_best=True, Descending_in_size=True)        
    WVD = OT.WVD #This is the resulting Weighted Voronoi Diagram, which contains an encoding of the optimal transport
    print("Centers: " +str(WVD.centers))
    print("Weights: " +str(OT.best_weights_so_far))
    print("Distance: " +str(OT.best_objective_so_far))
    
