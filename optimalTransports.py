'''
Created on Oct 6, 2019

@author: mvelasco
'''
import numpy as np
import pdb
import time


class Empirical_Measure:
    def __init__(self, data_points_vector, distance_fn):
        self.data_vectors = []
        self.distance=distance_fn
        self.ndata_vectors = len(data_points_vector)
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

    def nearest_data_point_index(self, vector):
            distances = np.array([self.distance(vector, data_point) for data_point in self.data_vectors])
            k = np.where(distances == np.min(distances))[0]
            return(k[0])

    
    def classify_nearest(self, vectors_to_classify):
        Classified_Points = [[] for k in range(self.ndata_vectors)]
        for vector in vectors_to_classify:
            k = self.nearest_data_point_index(vector)
            Classified_Points[k].append(vector)
        return Classified_Points

def dist(x,y):
    #For now uses the l2-norm but one can put here any norm
    assert(len(x)==2)
    assert(len(x)==len(y))
    return np.linalg.norm(x-y)    

def sample_q(numSamples):
    #Uniform distribution in [-1,1], [-1,1]
    ResultsArray = []
    Xs = np.random.uniform(-1,1,numSamples)
    Ys = np.random.uniform(-1,1,numSamples)
    for k in range(numSamples):
        ResultsArray.append([Xs[k],Ys[k]])
    return ResultsArray

class optimalTransportFinder:
    def __init__(self, empirical_data_points, sample_q_fn, distance_fn, num_MC = 1000):
        self.empirical_measure = Empirical_Measure(empirical_data_points, distance_fn)
        self.currentLambdas = np.zeros(len(empirical_data_points))
        self.bestLambdas = self.currentLambdas
        self.num_MC = num_MC
        self.sample_q_fn = sample_q_fn
        self.distance = distance_fn
        self.compute_current_state_vector()
        self.bestPsi = self.current_Psi_scalar_value
    
    def weighted_nearest_data_point_index(self, vector):
            N = self.empirical_measure.ndata_vectors
            weighted_distances = np.array([self.distance(vector, self.empirical_measure.data_vectors[k])-self.currentLambdas[k] for k in range(N)])
            k = np.where(weighted_distances == np.min(weighted_distances))[0]
            return(k[0])

    def weighted_classify_nearest(self, vectors_to_classify):
        Classified_Points = [[] for k in range(self.empirical_measure.ndata_vectors)]
        for vector in vectors_to_classify:
            k = self.weighted_nearest_data_point_index(vector)
            Classified_Points[k].append(vector)
        return Classified_Points

    def evaluate_phi_lambda_at_vector(self, vector):
        number_Dta_Points = self.empirical_measure.ndata_vectors
        weighted_distances = np.array([self.distance(vector, self.empirical_measure.data_vectors[k])-self.currentLambdas[k] for k in range(number_Dta_Points)])
        return np.min(weighted_distances)

    def evaluate_min_distance_at_vector(self, vector):
        number_Dta_Points = self.empirical_measure.ndata_vectors
        distances = np.array([self.distance(vector, self.empirical_measure.data_vectors[k]) for k in range(number_Dta_Points)])
        return np.min(distances)

    def compute_current_state_vector(self):
        N = self.empirical_measure.ndata_vectors        
        samples_q = self.sample_q_fn(self.num_MC)
        classified_samples = self.weighted_classify_nearest(samples_q)
        #assert(len(classified_samples)==N)
        #First we compute the super-gradient
        counts = [(-1)*len(res)/(self.num_MC) + (1/N) for res in classified_samples]
        self.current_super_gradient_value = np.array(counts)
        #Alternative computation of Psi
        #evaluated_samples = [self.evaluate_phi_lambda_at_vector(vector) for vector in samples_q]
        #alternativePsi = np.sum(evaluated_samples)

        #Computation of Psi
        weighted_evaluated_distances = []
        for k in range(N):
            for vector in classified_samples[k]:
                weighted_evaluated_distances.append(self.distance(self.empirical_measure.data_vectors[k], vector)-self.currentLambdas[k])
        
        self.current_Psi_scalar_value = np.sum(weighted_evaluated_distances)/self.num_MC



    def new_one_step(self, step_size):
        self.compute_current_state_vector()
        g = self.current_super_gradient_value        
        self.currentLambdas = self.currentLambdas + step_size*g
        #assert(np.abs(np.sum(self.currentLambdas))< 1e-8)
        #assert(np.abs(self.current_Psi_scalar_value-self.current_Psi_value())<1e-2)
        newPsi = self.current_Psi_scalar_value
        if newPsi >= self.bestPsi:
            self.current_super_gradient_norm = np.linalg.norm(self.current_super_gradient_value)
            print()
            print("Step "+str(1/step_size))
            print("Step Size "+str(step_size))
            print("Current Psi: "+str(self.current_Psi_scalar_value))
            print("Current norm: "+ str(self.current_super_gradient_norm))
            self.bestPsi = newPsi
            self.bestLambdas = self.currentLambdas

    
    def compute_optimal_lambda(self, numSteps, epsilon = 1.0):
        for k in range(numSteps):
            OT.new_one_step(epsilon/(k+1))        
        
        print()
        print("Computation of "+str(numSteps)+" steps completed.")
        self.print_Results()
        OT.currentLambdas = OT.bestLambdas

    def print_Results(self):
        print("Best Lower bound for W(q,mu): "+ str(OT.bestPsi))
        print("Gradient norm at optimality: "+str(self.current_super_gradient_norm))
        print("Best Lambdas achieving it: ")
        print(OT.bestLambdas)
        


if __name__ == "__main__":

    #1. Empirical measures:        
    #empirical_data_points_vector = [[1,0], [0,0], [0,1]]
    empirical_data_points_vector = [np.random.uniform(-1,1,2) for k in range(60)]
    print(empirical_data_points_vector)
    mu = Empirical_Measure(empirical_data_points_vector, dist)
    classify_data_points_vector = [[1,0], [0,0], [0,1], [1.1,0], [0,0.1], [0,1.1]]
    print(mu.classify_nearest(classify_data_points_vector))
    #2. optimal transport between q and a given empirical measure built from MonteCarlo
    t0 = time.time()
    OT = optimalTransportFinder(empirical_data_points_vector, sample_q, dist, 10000)
    OT.compute_optimal_lambda(20)
    t1 = time.time()
    print("Time elapsed: "+str(t1-t0))
    print("Step one finished")
    OT.compute_optimal_lambda(20, epsilon = 0.1)
    t2 = time.time()
    print("Step two finished")
    print("Time elapsed: "+str(t2-t1))
    
    
