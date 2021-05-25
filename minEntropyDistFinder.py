'''
Created on May 14, 2021

@author: mvelasco
'''
import numpy as np
import matplotlib.pyplot as plt
from optimalTransports import Empirical_Measure,Probability_Measure, Weighted_Voronoi_Diagram, dist,two_d_uniform_density,two_d_uniform_sample_q
from gurobipy import *
import pdb


class Polytope:
    """
    This class is a description of a polytope via inequalities. 
    It can compute the Chebyshev center of any such polytope
    

    The inequalities take the form
    Ineq_Vector \dot variables <= RHS
    and Eq_vectors \dot variables == RHS 

    The Chebyshev center inequalities require the dual norm of the constraints so the polytope requires a dualnorm_fn.
    
    """
    def __init__(self, dualnorm_fn):
        self.numIneqs = 0
        self.model = Model("Chebyshev_Center")
        self.dim = 0
        self.Ineq_vectors = []
        self.Ineq_RHss = []
        self.Eq_vectors = []
        self.Eq_RHss = []

        self.dualnorm_fn = dualnorm_fn
        self.gurobiVars = []
        self.XVars = []
        
    def initialize_cube(self, dim, diamK):
        """Creates a cube in R^dim dimensions, 
        with center (0,0) and infinity norm at most diamK
        slashed by \sum \lambda_i=0
        which we can use as initial setup """
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
        
        #Additional equality saying that the sum of the \lambda_i must be zero
        self.Eq_vectors.append(np.array([1.0 for k in range(dim)]))
        self.Eq_RHss.append(0.0)

        
    def initialize_chebyshev_model(self):
        """This function specifies the optimization problem to be run for finding Chebyshev centers
        
        """
        #One variable per dimension 
        names = ["X_"+str(k) for k in range(self.dim)]
        for name in names:
            self.gurobiVars.append(self.model.addVar(name=name,vtype=GRB.CONTINUOUS, lb = (-1)*float("inf"), ub = float("inf")))
        
        #additional nonnegative variable r, for the radius of the ball
        
        rvar = self.model.addVar(name="r",vtype=GRB.CONTINUOUS) #Gurobi DEFAULT behavior is making continuous variables automatically nonnegative
        self.gurobiVars.append(rvar)
        self.gurobiR = rvar
        #r will be the last variable

        #We construct the inequalities of the Tchebyshev center problem
        newIneq_vectors = []
        for vector in self.Ineq_vectors:
            newVector = np.zeros(len(vector)+1)
            
            for k in range(self.dim +1):
                if k < self.dim:
                    newVector[k] = vector[k]
                else:
                    newVector[k] = self.dualnorm_fn(vector)
            newIneq_vectors.append(newVector)

        #Next we add the inequalities to the Model:            
        for k in range(len(self.Ineq_vectors)):
            coeff_vector = newIneq_vectors[k]            
            try: 
                assert(len(coeff_vector)==len(self.gurobiVars))
            except:
                pdb.set_trace()
            gurobiLH = LinExpr(coeff_vector, self.gurobiVars)
            rhs = self.Ineq_RHss[k]
            self.model.addConstr(gurobiLH, sense = "<=", rhs = rhs, name = "Ineq_"+str(k))
        #The equalities involve only the X vars and not r
        self.XVars =[self.gurobiVars[k] for k in range(self.dim)]
        for k in range(len(self.Eq_vectors)):
            coeff_vector = self.Eq_vectors[k]
            gurobiLH = LinExpr(coeff_vector, self.XVars)
            rhs = self.Eq_RHss[k]
            self.model.addConstr(gurobiLH, sense = "==", rhs = rhs, name = "Eq_"+str(k))

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



class Two_Dim_Aux_Solver:
    """
        Given: 
        -An empirical measure
        -A probability measure
        -A radius delta>=0
        -A distance function
        -A number num_MC (how many MonteCarlo samplings are taken at each step)
        -A vector of current_weights (one per data point of the empirical measure)
                
        The class: Creates a weighted Voronoi-diagram and solves the 2-diml maximization problem of Theorem 1.3 - (1)
        via projected gradient descent (see do_projected_gradient_descent)
        
        After the optimization, the optimal points (u,v) are stored in:
        -The vector self.current_uv  (2-diml)       
        
        WARNING: The current implementation of projected gradient descent is quite naive and requires 
        some manual parameter tuning (choice of step size and number of steps)
    """
    def __init__(self, delta,  empirical_measure, probability_measure, distance_fn, num_MC, current_weights):
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
        self.best_gradient_norm = np.linalg.norm(self.compute_gradient())
        
    def compute_objective(self):
        u = self.current_uv[0]
        v = self.current_uv[1]
        value = -u-self.delta*v
        #Create a sample from our probability measure
        sample_vectors_array = self.probability_measure.sample_q(self.num_MC)
        #Evaluate the minimum weighted distance of each sampled point to the centers
        min_distances = self.WVD.compute_array_minimal_weighted_distances(sample_vectors_array)
        values_vector = np.exp((-1)*(v*min_distances + u + 1))
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
    
    def do_projected_gradient_descent(self,NumSteps, InitialStepSize, keep_track_of_best = False, step_type = "backtracking", verbose=False):
        assert(step_type in ["descending", "backtracking"])
        for k in range(NumSteps):
            self.gradient = self.compute_gradient()
            gradient = self.gradient
            if step_type == "descending":    
                CurrStepSize = InitialStepSize/(1+np.sqrt(k))#
            elif step_type == "backtracking":                
                #Backtracking implementation
                objective = self.compute_objective()
                tau = 0.5
                c = 0.5
                m = np.power(np.linalg.norm(gradient),2)
                alpha = InitialStepSize
                t = c*m
                candidate_future_objective = objective
                initial_position = self.current_uv
                j=0
                #
                while candidate_future_objective < objective + alpha*t and j < 20:
                    self.current_uv = initial_position + alpha * gradient
                    self.current_uv[1] = max(0.0, self.current_uv[1])
                    candidate_future_objective = self.compute_objective()
                    if verbose:
                        print("____________________________________________")
                        print("Backtracking step "+str(j))
                        print("Backtracking step size alpha: "+str(alpha))
                        print("Candidate_future_objective: "+str(candidate_future_objective))
                        print("Current objective: "+str(objective))
                        print("____________________________________________")

                    j=j+1
                    alpha = tau * alpha                                        
                    
                self.current_uv = initial_position
                CurrStepSize = alpha
            else:
                CurrStepSize = InitialStepSize
                
            new_uv = self.current_uv + CurrStepSize * gradient #gradient step
            new_uv[1] = max(0.0, new_uv[1]) #projection step
            self.current_uv = new_uv

            self.objective = self.compute_objective()
            self.grad_norm = np.linalg.norm(gradient)
            print("Two-diml gradient ascent step "+str(k)+":\n")
            self.print_current_status()
            if self.objective > self.best_objective_so_far:
                self.best_objective_so_far = self.objective
                self.best_uv = new_uv
                self.best_gradient_norm = self.grad_norm

        #if we keep track of the best then we should set it as the chosen weights
        if keep_track_of_best:
            self.current_uv = self.best_uv
        else:
            self.best_uv = self.current_uv
            self.best_gradient_norm = self.grad_norm
            
    def print_current_status(self):
        print("_____________________________________________________________")
        print("Curr_obj: " + str(self.objective)+ "")
        print("Curr_grad_norm: " + str(self.grad_norm)+ "")
        print("Current gradient: "+str(self.gradient))
        print("_____________________________________________________________")
        
class Minimum_Cross_Entropy_Dist_Finder:
    def __init__(self, delta,  diamK, empirical_measure, probability_measure, distance_fn, dualnorm_fn, num_MC, num_UVsteps, UV_steptype, UV_initial_stepsize):
        """Given:
            -A radius delta > 0
            -An empirical measure \mu
            -A probability distribution p
            -A distance function
            -A dualnorm function (for the Chebyshev problem)
            -The num_MC parameter
            -An initial diameter diamK>0 (the infinity norm of the support set K suffices)
            
            the class computes a vector of weights \lambda^* and a pair (u^*,v^*) such that
            q*exp(-1-v\phi_{\lambda}(x)-u) is a good approximation to the probability distribution 
            with minimum cross entropy in the Wasserstein ball of radius delta centered at \mu 
            
        """

        self.num_MC = num_MC
        assert(isinstance(empirical_measure,Empirical_Measure ))
        self.empirical_measure = empirical_measure
        assert(isinstance(probability_measure,Probability_Measure ))
        self.probability_measure = probability_measure

        self.distance_fn = distance_fn

        self.distance_fn = self.distance_fn
        self.delta = delta
        self.diamK = diamK
        self.dualnorm_fn = dualnorm_fn

        #Start with a cube
        self.current_polytope = Polytope(dualnorm_fn)
        N = self.empirical_measure.ndata_vectors
        self.current_polytope.initialize_cube(N, diamK)
        self.current_polytope.initialize_chebyshev_model()
        self.current_polytope.compute_chebyshev_center()

        #Create a solver for the auxiliary two-diml problem
        self.current_lambdas = self.current_polytope.current_Chebyshev_Center
        self.TDS = Two_Dim_Aux_Solver(
            delta, 
            self.empirical_measure, 
            self.probability_measure, 
            self.distance_fn, 
            self.num_MC, 
            self.current_lambdas)
        
        #Create the samples of the prior
        self.maxUVsteps = num_UVsteps
        assert(UV_steptype in ["descending", "backtracking"])
        self.UVInitialStepType = UV_steptype
        self.UVInitialStepSize = UV_initial_stepsize


    def do_lambda_star_one_step(self, verbose = False):
        #Do a projected gradient descent epoch:
        self.current_polytope.compute_chebyshev_center()        
        self.current_lambdas = self.current_polytope.current_Chebyshev_Center
        self.TDS.WVD.weights = self.current_lambdas
        self.TDS.do_projected_gradient_descent(self.maxUVsteps,self.UVInitialStepSize,keep_track_of_best=False)
        self.current_lambdas = self.TDS.WVD.weights
        UVvector = self.TDS.best_uv 
        self.best_uv = UVvector
        #We add a new cutting hyperplane to our polytope_
        self.samples_p = self.probability_measure.sample_q(10*self.num_MC)
        gradient = self.TDS.WVD.compute_gradient_from_distorted_averages_in_cells(self.samples_p, UVvector)
        rhs = np.dot(gradient, self.current_lambdas)
        #Add the new cutting hyperplane to our polytope
        self.current_polytope.new_linear_Ineq((-1)*gradient, (-1)*rhs)
        
        

    def do_lambda_star_several_steps(self, numsteps, verbose=False):        
        for k in range(numsteps):
            if verbose:
                print("Computation of Chebyshev center number "+str(k)+" of "+str(numsteps)+" :")
            self.do_lambda_star_one_step(verbose)
        #once it is done it does a final TDS with very high detail:
        print("Computing final round with higher res MonteCarlo")
        self.TDS.WVD.weights = self.current_lambdas
        MC_multiplier = 20
        UVSteps_multiplier = 2
        InitialStepSize_multiplier = 0.3
        self.TDS.num_MC = MC_multiplier*self.TDS.num_MC 
        self.TDS.do_projected_gradient_descent(UVSteps_multiplier*self.maxUVsteps,InitialStepSize_multiplier*self.UVInitialStepSize,keep_track_of_best=False)
        self.current_lambdas = self.TDS.WVD.weights
        UVvector = self.TDS.best_uv 
        self.best_uv = UVvector



    def plot_current_weights_dist(self, namestring, num_points = 500):        
        plt.figure(dpi=150)
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(1.0)
        M = num_points
        #We will use an MxM grid of points
        xvalues = np.linspace(0,1,num=M);
        yvalues = np.linspace(0,1,num=M);        
        xx, yy = np.meshgrid(xvalues, yvalues)
        UVvector = self.TDS.best_uv
        #Next we need use the formula for the minimum entropy distribution
        #It is the result of evaluating the density times the exponential distortion factos
        g = lambda a,b: self.probability_measure.evaluate_density_fn(np.array([a,b]))*self.TDS.WVD.uv_distorted_weight(np.array([a,b]), UVvector)
        Vg = np.vectorize(g)#Vectorization to be able to apply it to a numpy meshgrid
        z = Vg(xx,yy)
        plt.pcolormesh(xvalues, yvalues, z, cmap="RdYlBu_r")
        #Next we plot the centers in black
        em = self.empirical_measure
        Xs = [em.data_vectors[k][0] for k in range(em.ndata_vectors)]
        Ys = [em.data_vectors[k][1] for k in range(em.ndata_vectors)]
        plt.scatter(Xs, Ys, c="black", alpha=0.9) 
        #Finally we save the image and show it to screen
        plt.savefig(namestring)
        plt.show()



def norm(x):
    return np.linalg.norm(x)




def example_two_dim_aux_solver():
    """This small script shows how to instantiate the underlying Two-dimensional projected gradient descent solver"""
    N=5 #number of centers of the empirical measure
    centers_array = [np.random.uniform(-1,1,2) for k in range(N)] #Chosen uniformly in the square [-1,1]x[-1,1]
    #Deterministic centers array for article image:
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(two_d_uniform_sample_q, two_d_uniform_density)
    #First, we construct the Two-dimensional solver which carries out the gradient descent
    num_MC=10000
    initial_weights = np.zeros(empirical_measure.ndata_vectors)
    delta=0.05
    TDS = Two_Dim_Aux_Solver(delta, empirical_measure,probability_measure,dist,num_MC,initial_weights)
    obj = TDS.compute_objective()
    grad = TDS.compute_gradient()
    print(obj)
    print(grad)
    TDS.do_projected_gradient_descent(NumSteps = 100, StepSize=1.0)
    print(TDS.current_uv)
    print("Done")
    


if __name__ == "__main__":    
    #example_two_dim_aux_solver()
    #0.Weighted Voronoi diagrams:
    N=5 #number of centers of the empirical measure
    centers_array = [np.random.uniform(0,1,2) for k in range(N)] #Chosen uniformly in the square [-1,1]x[-1,1]
    empirical_measure = Empirical_Measure(centers_array)
    #The probability measure is specified by its sampling function
    probability_measure = Probability_Measure(two_d_uniform_sample_q, two_d_uniform_density)
    num_MC=1000    
    #Next we construct the full minimum entropy finder    
    diamK = 1.0
    delta = 0.05
    distance_fn = dist
    dualnorm_fn = norm
    num_MC = 9000
    num_UVsteps = 20
    UV_steptype = "backtracking"

    UV_initial_stepsize = 5
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
    MCE.do_lambda_star_several_steps(numsteps=2, verbose=True)
    #Results (which encode the minimum cross entropy distribution via q = p*exp(-1-v\phi_{\lambda}(x)-u)
    #where p is the density of the probability distribution above
    print("Final_uv: " + str(MCE.best_uv))
    print("Best_lambdas: "+str(MCE.current_lambdas))
    print("Min cross entropy: "+str(MCE.best_objective))
    print("")
    MCE.plot_current_weights_dist("intento.png", num_points=300)