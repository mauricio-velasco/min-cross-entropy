'''
Created on Oct 12, 2019

@author: mvelasco
'''
from optimalTransports import Empirical_Measure, dist
from gurobipy import *
import numpy as np
import pdb


class polytope:
    def __init__(self, dualnorm_fn, Ineq_vectors=[], Ineq_RHss=[], Eq_vectors=[], Eq_RHss=[]):
        self.numIneqs = 0
        self.model = Model("Chebyshev_Center")
        #Checks input quality
        self.dim = 0
        if len(Ineq_vectors)>0:
            self.dim = len(Ineq_vectors[0])
            for vector in Ineq_vectors:
                if len(vector)!= self.dim:
                    raise AssertionError("Inequality vectors have unequal sizes.")
            assert( len(Ineq_vectors)==len(Ineq_RHss)) 
            for vector in Eq_vectors:
                if len(vector)!= self.dim:
                    raise AssertionError("Equality vectors have unequal sizes.")
            assert( len(Eq_vectors)==len(Eq_RHss)) 
                
        self.Ineq_vectors = Ineq_vectors
        self.Ineq_RHss = Ineq_RHss
        self.Eq_vectors = Eq_vectors
        self.Eq_RHss = Eq_RHss
        self.dualnorm_fn = dualnorm_fn
        self.gurobiVars = []
        self.XVars = []
        
    def initialize_cube(self, dim, diamK):
        self.dim = dim
        for k in range(dim):
            vector = np.zeros(dim)
            vector[k]=1.0
            self.Ineq_vectors.append(vector)
            vector = np.zeros(dim)
            vector[k]=-1.0
            self.Ineq_vectors.append(vector)
            self.Ineq_RHss.append(diamK)
            self.Ineq_RHss.append(diamK)            
            self.numIneqs += 2

        self.Eq_vectors.append(np.array([1.0 for k in range(dim)]))
        self.Eq_RHss.append(0.0)
        
    def initialize_chebyshev_model(self):
        #Primero se construyen las variables, una por cada dimension y una adicional por el radio.
        #En el proceso especificamos la funcion objetivo
        names = ["X_"+str(k) for k in range(self.dim)]
        for name in names:
            self.gurobiVars.append(self.model.addVar(name=name,vtype=GRB.CONTINUOUS))
        #add nonnegative variable r:
        rvar = self.model.addVar(lb=0.0, name="r",vtype=GRB.CONTINUOUS)
        self.gurobiVars.append(rvar)
        self.gurobiR = rvar
        #Luego se describen las restricciones que tenemos
        newIneq_vectors = []
        for vector in self.Ineq_vectors:
            newVector = np.zeros(len(vector)+1)
            for k in range(self.dim +1):
                if k< self.dim:
                    newVector[k] = vector[k]
                else:
                    newVector[k] = self.dualnorm_fn(vector)
            newIneq_vectors.append(newVector)

        #We add the equalities            
        for k in range(len(self.Ineq_vectors)):
            coeff_vector = newIneq_vectors[k]
            gurobiLH = LinExpr(coeff_vector, self.gurobiVars)
            rhs = self.Ineq_RHss[k]
            self.model.addConstr(gurobiLH, sense = "<=", rhs = rhs, name = "Ineq_"+str(k))
        #The equalities involve only the X vars and not r
        self.XVars =[self.gurobiVars[k] for k in range(self.dim)]
        for k in range(len(self.Eq_vectors)):
            coeff_vector = self.Eq_vectors[k]
            gurobiLH = LinExpr(coeff_vector, self.XVars)
            rhs = self.Eq_RHss[k]
            self.model.addConstr(gurobiLH, sense = "==", rhs = rhs, name = "Ineq_"+str(k))

        #We specify the objective function and that it is a maximization problem,            
        obj_coeffs = [0.0 for k in range(self.dim)]
        obj_coeffs.append(1.0)
        gurobiOBJ = LinExpr(obj_coeffs, self.gurobiVars)
        self.model.setObjective(gurobiOBJ , sense = GRB.MAXIMIZE )
        self.model.update()

    def compute_chebyshev_center(self):
        self.model.update()
        self.model.optimize()
        self.current_Chebyshev_Center = np.array([Var.X for Var in self.XVars])
        self.current_r = self.gurobiR.X

    def new_linear_Ineq(self, coeffs_vector, rhs):
        #We add a new linear inequality to the polytope and to the model
        assert(self.dim == len(coeffs_vector))
        self.Ineq_vectors.append(np.array(coeffs_vector))
        self.Ineq_RHss.append(rhs)
        self.numIneqs += 1
        
        newVector = np.zeros(self.dim + 1)
        for k in range(self.dim):
            newVector[k] = coeffs_vector[k]
        newVector[self.dim] = self.dualnorm_fn(coeffs_vector)
        gurobiLH = LinExpr(newVector, self.gurobiVars)
        self.model.addConstr(gurobiLH, sense = "<=", rhs = rhs, name = "Ineq_"+str(self.numIneqs))
        
class minimum_cross_entropy_finder:
    def __init__(self, num_MC, samplep_fn, empirical_measure, delta, diamK,dualnorm_fn):
        self.num_MC = num_MC
        self.samplep_fn = samplep_fn
        self.empirical_measure = empirical_measure
        self.distance = self.empirical_measure.distance
        self.dim = self.empirical_measure.dim
        self.delta = delta
        self.diamK = diamK
        self.dualnorm_fn = dualnorm_fn
        #Start with a cube
        self.current_polytope = polytope(dualnorm_fn)
        N = self.empirical_measure.ndata_vectors
        self.current_polytope.initialize_cube(N, diamK)
        self.current_polytope.initialize_chebyshev_model()
        self.current_polytope.compute_chebyshev_center()
        #Create the samples of the prior
        self.samples_p = self.samplep_fn(self.num_MC)        
        #Begin at Chebyshev center
        self.current_lambdas = self.current_polytope.current_Chebyshev_Center
        self.maxUVsteps = 200

    def weighted_nearest_data_point_index(self, vector):
            N = self.empirical_measure.ndata_vectors
            weighted_distances = np.array([self.distance(vector, self.empirical_measure.data_vectors[k])-self.current_lambdas[k] for k in range(N)])
            k = np.where(weighted_distances == np.min(weighted_distances))[0]
            return(k[0])

    def weighted_classify_nearest(self, vectors_to_classify):
        Classified_Points = [[] for k in range(self.empirical_measure.ndata_vectors)]
        for vector in vectors_to_classify:
            k = self.weighted_nearest_data_point_index(vector)
            Classified_Points[k].append(vector)
        return Classified_Points

    def minimum_weighted_distance(self,vector):
        N = self.empirical_measure.ndata_vectors
        weighted_distances = [self.distance(vector,self.empirical_measure.data_vectors[k])-self.current_lambdas[k] for k in range(N)]
        return np.min(weighted_distances)

    def compute_good_uv(self):
        phi_lambdas = [self.minimum_weighted_distance(vector) for vector in self.samples_p]
        
        #Begin by computing the current initial value
        self.currentUV = np.array([0.0,0.0])
        self.currentUVGradient = np.zeros(2)
        u=self.currentUV[0]
        v=self.currentUV[1]        
        self.best_objective_value = -u-v*self.delta-np.sum([np.exp(-1-v*phiL-u) for phiL in phi_lambdas])/self.num_MC
        self.bestUV = self.currentUV

        #Increment by gradient ascent
        for k in range(self.maxUVsteps):            
            u=self.currentUV[0]
            v=self.currentUV[1]
            self.currentUVGradient[0] = -1 + (np.sum([np.exp(-1-v*phiL-u) for phiL in phi_lambdas]))/self.num_MC
            self.currentUVGradient[1] = -self.delta + (np.sum([phiL*np.exp(-1-v*phiL-u) for phiL in phi_lambdas]))/self.num_MC
            self.current_objective_value = -u-v*self.delta-np.sum([np.exp(-1-v*phiL-u) for phiL in phi_lambdas])/self.num_MC
            stepSize = (1/(self.delta+0.1))*(1/(np.log(k+1)+1))*self.dualnorm_fn(self.currentUVGradient)

            #En u siempre se hace un paso de descenso del gradiente
            self.currentUV[0]+=self.currentUVGradient[0]*stepSize
            #En v intentamos dar el paso, si nos salimos hay que proyectar            
            nextV = self.currentUV[1] + self.currentUVGradient[1]*stepSize
            if nextV<=0.0:
                self.currentUV[1] = 0.0
            else:
                self.currentUV[1] = nextV

            if self.current_objective_value>= self.best_objective_value:
                self.best_objective_value = self.current_objective_value
                self.bestUV = self.currentUV
                print(k)
                print(self.current_objective_value)
                print(self.currentUV)
                print(self.currentUVGradient)
    
    def compute_separating_hyperplane(self):
        N = self.empirical_measure.ndata_vectors        
        classified_samples = self.weighted_classify_nearest(self.samples_p)
        #First we compute the super-gradient
        counts = [(-1)*len(res)/(self.num_MC) + (1/N) for res in classified_samples]
        return np.array(counts)
        
        
    def cutting_plane_one_step(self):
        self.compute_good_uv()
        g = self.compute_separating_hyperplane()
        rhs = (-1)*np.dot(g,self.current_lambdas)
        self.current_polytope.new_linear_Ineq((-1)*g, rhs)
        self.current_polytope.compute_chebyshev_center()
        self.current_lambdas = self.current_polytope.current_Chebyshev_Center
        print(self.current_polytope.current_r)
        print(self.currentUV)
        print(self.current_lambdas)

def norm(x):
    return np.linalg.norm(x)

#This is the prior, implemented as a function capable of producing samples
def sample_p(numSamples):
    #Uniform distribution in [-1,1], [-1,1]
    ResultsArray = []
    Xs = np.random.uniform(-1,1,numSamples)
    Ys = np.random.uniform(-1,1,numSamples)
    for k in range(numSamples):
        ResultsArray.append([Xs[k],Ys[k]])
    return ResultsArray


if __name__ == "__main__":


    P = polytope(norm)
    N=10
    diamK=3.0
    P.initialize_cube(N,diamK)
    P.initialize_chebyshev_model()
    P.compute_chebyshev_center()
    print(P.current_Chebyshev_Center)
    print(P.dim)
    print(P.numIneqs)
    print("done")

    #empirical_data_points_vector = [np.random.uniform(-1,1,2) for k in range(10)]
    empirical_data_points_vector = [[1,0], [0,0], [0,1]]
    mu = Empirical_Measure(empirical_data_points_vector, dist)
    MF = minimum_cross_entropy_finder(10000, sample_p, mu, 0.05, 2.0, norm)
    MF.cutting_plane_one_step()