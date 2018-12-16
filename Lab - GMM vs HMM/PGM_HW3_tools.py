import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from scipy.spatial.distance import cdist

################################################################################################

class my_KMeans():
    """
    Class for K-means clustering
    """
    def __init__(self,n_clusters,RandomState=42,max_iter=200):
        '''
        Parameters & Attributes:
        
        n_clusters: integer
            number of clusters to form
        RandomState: interger
            determines random number generation for centroid initialization
        centers: np.array
            array containing the centroids of the clusters
        init_centers_: np.array
            array containing the initial centroids
        labels_: (n,) np.array
            array containing labels of each point
        distortion_: np.array
            array containing the sum of squared distances of samples to their closest centroid.
        max_iter: float
            maximum iterations for the convergence
        '''
        self.k_ = n_clusters
        self.RandomState_ = RandomState
        self.max_iter_ = max_iter
        self.centers = None
        self.init_centers_ = None
        self.labels_ = None
        self.distortion_ = None
        
       
    def fit(self, X):
        """ Generate the centroids
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        """
        # Initialization of centroids
        rng=np.random.RandomState(self.RandomState_)
        self.init_centers_ = X[rng.choice(range(X.shape[0]),self.k_,replace=False)]
        self.init_labels_ = np.argmin(cdist(X,self.init_centers_),axis=1)
        self.init_distortion_ = np.sum(np.linalg.norm(X-self.init_centers_[self.init_labels_],axis=1)**2)
               
        # Parameters
        convergence = False
        self.centers = self.init_centers_.copy()
        self.labels_ = self.init_labels_.copy()
        distortion=self.init_distortion_.copy()
        it=0
        
        # Convergence
        while (not(convergence) and it<self.max_iter_):
            it+=1
            for k in range(self.k_): self.centers[k]=np.mean(X[self.labels_==k],axis=0)
            self.labels_ = np.argmin(cdist(X,self.centers),axis=1)
            self.distortion_=np.sum(np.linalg.norm(X-self.centers[self.labels_],axis=1)**2)
            
            if np.abs(distortion-self.distortion_)<1e-3:
                convergence=True
            
            centers=self.centers
            distortion=self.distortion_ 
        self.it_=it
            
################################################################################################
################################################################################################
    
class my_GMM():
    """ 
    Gaussian Mixture Model with general covariances matrices 
    """   
    def __init__(self, n_clusters, iter_max=100, tol=1e-5, RandomState=24):
        '''
        Parameters & Attributes:
        
        k_: integer
            number of clusters
        mu_: np.array
            array containing means
        Sigma_: np.array
            array containing covariance matrices
        tau_: (n, K) np.array
            conditional probabilities for all data points "p(z/x)"
        labels_: (n, ) np.array
            labels for data points
        pi_: (K,) np.array
            array containing parameters for the multinomial latent variables
        iter_max: float
            maximum iterations allowed for the EM convergence
        tol: float
            tolerence for checking the EM convergence
        RandomState: int
            determines random number generation for centroid initialization
        '''
        self.k_ = n_clusters
        self.mu_ = None
        self.Sigma_ = None
        self.tau_ = None
        self.labels_ = None
        self.pi_ = None
        self.iter_max_ = iter_max
        self.tol_ = tol
        self.RandomState=RandomState

    def compute_tau(self, X, mu, Sigma):
        '''Compute the conditional probability matrix p(z/x)
        shape: (n, K)
        '''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))          
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).T
        return pdf_k_s*self.pi_/((np.sum(pdf_k_s*self.pi_,axis=1))[:,None])

    def E_step(self, X, mu, Sigma, cond_prob):
        '''Compute the expectation of the complete loglikelihood to check increment'''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).reshape(self.k_,n).T
        return np.sum(cond_prob*np.log(pdf_k_s) + cond_prob*np.log(self.pi_))
    
    def compute_complete_likelihood(self, X, labels):
        """ Compute the complete likelihood for a given labels 
        
        Parameters:
        -----------
         X: (n, p) np.array
            Data matrix
        labels: (n, ) np.array
            Data labels
        
        Returns:
        -----
        complete likelihood       
        """        
        return self.E_step(X,self.mu_,self.Sigma_,np.eye(self.k_)[labels])    
        
    def fit(self, X):
        """ Find the parameters mu_ and Sigma_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        """
        n=X.shape[0]
        p=X.shape[1]        
            
        # Initialization with kmeans   
        k_init=my_KMeans(n_clusters=self.k_,RandomState=self.RandomState)
        k_init.fit(X)
        self.labels_=k_init.labels_
        self.pi_=np.unique(self.labels_,return_counts=True)[1]/n
        self.mu_=k_init.centers
        self.Sigma_=[np.matmul((X[k_init.labels_==k]-self.mu_[k]).T,(X[k_init.labels_==k]-self.mu_[k])/(n*self.pi_[k])) for k in range(self.k_)]            
        
        converged=False
        it=0
        
        #First E-Step
        self.tau_ = self.compute_tau(X,self.mu_,self.Sigma_)
        En=self.E_step(X,self.mu_,self.Sigma_,self.tau_)
        
        
        while ((not converged) and it<self.iter_max_):
            #M-Step
            self.pi_=np.mean(self.tau_,axis=0)
            self.mu_=np.matmul(self.tau_.T,X)/(np.sum(self.tau_,axis=0).reshape(-1,1))
            self.Sigma_=np.array([np.matmul((X-self.mu_[k]).T,((X-self.mu_[k])*self.tau_[:,k].reshape(-1,1)))/(np.sum(self.tau_[:,k])) for k in range(self.k_)])
            
            #E-Step
            Enp1=self.E_step(X,self.mu_,self.Sigma_,self.tau_)
            if (np.abs(Enp1/En-1)) < self.tol_:
                converged=True
            it+=1
            En=Enp1
            self.tau_ = self.compute_tau(X,self.mu_,self.Sigma_)
        
        # Assigning labels
        self.labels_=np.argmax(self.tau_,axis=1)  
        
        # Computing complete likelihood
        self.complete_likelihood_ = self.compute_complete_likelihood(X,self.labels_)
        
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        return np.argmax(self.compute_tau(X,self.mu_,self.Sigma_),axis=1)
    

    def predict_proba(self, X):
        """ Predict probability vector for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        proba: (n, k) np.array        
        """
        return self.compute_tau(X,self.mu_,self.Sigma_)
    
########################################################################################### 
###########################################################################################

class my_HMM():
    """ 
    Hidden Markov Model
    """   
    def __init__(self, n_clusters, initialization="GMM", iter_max=100, tol=1e-5, RandomState=24):
        '''
        Parameters & Attributes:
        
        k_: integer
            number of clusters
        initialization: str
            type of initialization (GMM for Gaussian Mixture Model initialization and KMeans for Kmeans initialization)
        iter_max: float
            maximum iterations allowed for the EM convergence
        tol: float
            tolerence for checking the EM convergence
        RandomState: int
            determines random number generation for centroid initialization    
        
        mu_: np.array
            array containing means
        Sigma_: np.array
            array containing covariance matrices
        tau_t_i_: (n, K) np.array
            conditional probability p(q_t/u1,..,uT)
        tau_t_ij_: (n, K) np.array
            conditional probability p(q_t+1,q_t/u1,..,uT)
        labels_: (n, ) np.array
            labels for data points
        pi_: (K,) np.array
            array containing parameters for the multinomial latent variables
        A_: (K, K) np.array
            transition matrix 
        alpha_: (T,K)
            alpha message matrix
        beta_: (T,K) 
            beta message matrix
        pdf_k_s: (T,K) np.array
            normal densities applied to each observation for each cluster
        complete_likelihood_: float
            maximum complete loglikelihood for the training data
        uncomplete_likelihood_: float
            maximum uncomplete loglikelihood for the training data       
        '''
        self.k_ = n_clusters
        self.initialization = initialization
        self.iter_max_ = iter_max
        self.tol_ = tol
        self.RandomState=RandomState
        
        self.mu_ = None
        self.Sigma_ = None
        self.tau_t_i_ = None
        self.tau_t_ij_ = None
        self.labels_ = None
        self.pi_ = None
        self.A_ = None
        self.alpha_ = None
        self.beta_ = None
        self.pdf_k_s = None
        self.complete_likelihood_ = None
        self.uncomplete_likelihood_ = None
        
    
    def compute_vector_with_logs(self, V, log=False):
        '''
        Compute sum of a vector V with logs
        
        Parameters:
        -----------
        V: (n,) np.array
            vector
        
        Returns:
        -----
        log of the sum of the vector : float
        '''
        if not log:
            V = np.log(V)
        
        max_log = np.max(V)
        
        return max_log + np.log(np.sum(np.exp(V-max_log)))
    
    def compute_matrix_with_logs(self, M, log=False, axis=1):
        '''
        Compute sum of vectors in a matrix M with logs
        
        Parameters:
        -----------
        M: (n,p) np.array
            matrix
        
        Returns:
        -----
        log of the sum of matrix vectors : np.array (n,)
        '''
        if not log:
            M = np.log(M)
        
        max_log = np.max(M, axis=axis)    
        if axis==1:
            max_log=max_log.reshape(-1,1)
            return max_log + np.log(np.sum(np.exp(M-max_log), axis=axis).reshape(-1,1))
        
        return max_log + np.log(np.sum(np.exp(M-max_log), axis=axis))
    
    
    def alpha(self, X, A, pi, pdf_k_s):
        '''
        Alpha recursion
        
        Parameters:
        -----------
        X: (T,P) np.array
            data matrix
        A: (K, K) np.array
            transition matrix
        pi: (K,) np.array
            multinomial parameters
        pdf_k_s: (T,K) np.array
            normal densities applied to each observation for each cluster           
        
        Returns:
        -----
        alpha messages : np.array (T,K)
        '''
        alpha_M = np.zeros((X.shape[0], self.mu_.shape[0]))        
        for t in range(alpha_M.shape[0]):
            for qt in range(alpha_M.shape[1]):
                if t==0:
                    alpha_M[t,qt] = np.log(multivariate_normal(self.mu_[qt],self.Sigma_[qt]).pdf(X[0])) + np.log(pi[qt])
                else:
                    logs_vector=np.log(A[qt]) + alpha_M[t-1]
                    alpha_M[t,qt] = np.log(pdf_k_s[t,qt]) + self.compute_vector_with_logs(logs_vector, log=True)
        return alpha_M

    def beta(self, X, A, pdf_k_s):
        '''
        Beta recursion
        
        Parameters:
        -----------
        X: (T,P) np.array
            data matrix
        A: (K, K) np.array
            transition matrix
        pdf_k_s: (T,K) np.array
            normal densities applied to each observation for each cluster
        Returns:
        -----
        beta messages : np.array (T,K)
        '''
        beta_M = np.zeros((X.shape[0], self.mu_.shape[0]))
        for t in range(beta_M.shape[0])[::-1]:
            for qt in range(beta_M.shape[1])[::-1]:
                if t==X.shape[0]-1:
                    beta_M[t,qt] = 0
                else:
                    logs_vector = np.log(A[:,qt]) + np.log(pdf_k_s[t+1]) + beta_M[t+1]
                    beta_M[t,qt]=self.compute_vector_with_logs(logs_vector, log=True)
        return beta_M
    
    def viterbi(self, X, A, pi):
        '''
        Execute the viterbi algorithm to find the most probable sequence of states
        
        Parameters:
        -----------
        X: (T,P) np.array
            data matrix
        A: (K, K) np.array
            transition matrix
        pi: (K,) np.array
            multinomial parameters
            
        Returns:
        -----
        (alpha messages, most probable states sequence) : (np.array (T,K), np.array(T,))
        '''
        pdf_k_s = self.compute_normals(X)
        alpha_M = np.zeros((X.shape[0], self.mu_.shape[0]))
        argmaxs = np.zeros((X.shape[0],self.mu_.shape[0]))
        labels = np.zeros(X.shape[0], dtype=int)
        for t in range(alpha_M.shape[0]):
            for qt in range(alpha_M.shape[1]):
                if t==0:
                    alpha_M[t,qt] = np.log(multivariate_normal(self.mu_[qt],self.Sigma_[qt]).pdf(X[0])) + np.log(pi[qt])
                else:
                    alpha_M[t,qt] = np.log(pdf_k_s[t,qt]) + np.max((np.log(A[:,qt]) + alpha_M[t-1,:]))
                argmaxs[t] = np.argmax(alpha_M[t],axis=0)
        
        labels[-1] = np.max(argmaxs[-1])
        for t in range(2,X.shape[0]+1):
            labels[-t] = argmaxs[-t,labels[-t+1]]
        
        return labels

    def tau_t_i(self, alpha, beta):
        '''
        Compute the conditional probability p(q_t/u1,..,uT)
        
        Parameters:
        -----------
        alpha: (T, K) np.array
            alpha messages matrix
        beta: (T, K) np.array
            beta messages matrix
        
        Returns:
        -----
        Conditional probabilities: np.array (T,K)
        '''        
        return np.exp(alpha + beta - self.compute_matrix_with_logs(alpha+beta, log=True, axis=1))    

    def tau_t_ij(self, alpha, beta, pdf_k_s, A):
        '''
        Compute the conditional probability p(q_t+1, q_t/u1,..,uT)
        
        Parameters:
        -----------
        alpha: (T, K) np.array
            alpha messages matrix
        beta: (T, K) np.array
            beta messages matrix
        pdf_k_s: (T,K) np.array
            normal densities applied to each observation for each cluster
        A: (K, K) np.array
            transition matrix
        
        Returns:
        -----
        Conditional probabilities: np.array (T,K,K)  
        '''
        p_ys = self.compute_vector_with_logs(alpha[0]+beta[0], log=True)
        alpha = alpha[:-1]
        beta = beta[1:]
        pdf_k_s = pdf_k_s[1:]

        return np.exp( ((beta + np.log(pdf_k_s))[:,:,None] + np.log(A)) + alpha[:,None,:] - p_ys)
   
    def compute_normals(self, X):
        '''
        Compute the the normal density function for the vectors of X
        with the parameters self.mu_ and self.Sigma_
        
        Parameters:
        -----------
        X: (T, p) np.array
            Data matrix
        
        Returns:
        -----
        Normal densities: np.array (T,K)   
        '''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(self.mu_[k],self.Sigma_[k]).pdf(x))               
        return np.array([pdf_k(k,X) for k in range(self.k_)]).reshape(self.k_,n).T 
    
    def update_tau(self, X):       
        '''
        Update alpha, beta and the normal densities to update the conditional probabilities (tau)
        
        Parameters:
        -----------
        X: (T, p) np.array
            Data matrix
            
        Returns:
        -----
        None   
        '''
        self.pdf_k_s=self.compute_normals(X)  
        self.alpha_ = self.alpha(X, self.A_, self.pi_, self.pdf_k_s)
        self.beta_ = self.beta(X, self.A_, self.pdf_k_s)
        self.tau_t_i_ = self.tau_t_i(self.alpha_, self.beta_)
        self.tau_t_ij_ = self.tau_t_ij(self.alpha_, self.beta_, self.pdf_k_s, self.A_)
    

    def E_step(self, X, A, pi, pdf_k_s, tau_t_i, tau_t_ij):
        '''
        Compute the expectation of the complete loglikelihood to check increment
        
        Parameters:
        -----------
        X: (T, p) np.array
            Data matrix
        A: (K, K) np.array
            Transition matrix
        pi: (K,) np.array
            multinomial parameters
        pdf_k_s: (T,K) np.array
            normal densities applied to each observation for each cluster
        tau_t_i: (T,K) np.array
            conditional probability p(q_t/u1,..,uT)
        tau_t_ij: (T,K,K) np.array
            conditional probability p(q_t+1, q_t/u1,..,uT)
        
        Returns:
        -----
        Expectation: float      
        '''    
        E = np.sum(tau_t_ij*np.log(A))
        E += np.dot(tau_t_i[0], np.log(pi))
        E += np.sum(tau_t_i * np.log(pdf_k_s))        
        return E
    
    def compute_complete_likelihood(self, X, labels):
        '''
        Compute the complete likelihood for a given labels 
        
        Parameters:
        -----------
         X: (T, p) np.array
            Data matrix
        labels: (T, ) np.array
            Data labels
        
        Returns:
        -----
        complete likelihood: float
        '''              
        one_hot=np.eye(self.k_)[labels]
        f=lambda t : np.matmul(one_hot[t+1][:,None],one_hot[t][None,:])
        f_vec = np.vectorize(f, signature='()->(n,n)')
        
        return self.E_step(X,self.A_,self.pi_, self.compute_normals(X), one_hot, f_vec(np.arange(X.shape[0]-1)))    
        
    def fit(self, X):
        '''
        Find the parameters mu_, Sigma_, pi_ and A_
        that better fit the data
        
        Parameters:
        -----------
        X: (T, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        '''
        n=X.shape[0]
        p=X.shape[1]        
            
        # Initialization with kmeans 
        if self.initialization=="KMeans":
            k_init=my_KMeans(n_clusters=self.k_,RandomState=self.RandomState)
            k_init.fit(X)
            self.labels_=k_init.labels_
            self.pi_=np.unique(self.labels_,return_counts=True)[1]/n
            self.mu_=k_init.centers
            self.Sigma_=[np.matmul((X[k_init.labels_==k]-self.mu_[k]).T,(X[k_init.labels_==k]-self.mu_[k])/(n*self.pi_[k])) for k in range(self.k_)]            
            self.A_ = np.ones((self.k_,self.k_))/4
        
        # Initialization with GMM 
        if self.initialization=="GMM":
            GMM = my_GMM(n_clusters=self.k_, RandomState=self.RandomState)
            GMM.fit(X)
            self.labels_=GMM.labels_
            self.pi_=GMM.pi_
            self.mu_=GMM.mu_
            self.Sigma_=GMM.Sigma_
            self.A_ = np.ones((self.k_,self.k_))/4
        
        converged=False
        it=0
        
        #First E-Step
        self.update_tau(X)        
        En=self.E_step(X,self.A_,self.pi_, self.pdf_k_s, self.tau_t_i_, self.tau_t_ij_)
        
        while ((not converged) and it<self.iter_max_):
            #M-Step
            self.pi_= np.exp(np.log(self.tau_t_i_[0]) - self.compute_vector_with_logs(self.tau_t_i_[0]))        
            self.mu_=np.matmul(self.tau_t_i_.T,X)/ np.exp(self.compute_matrix_with_logs(self.tau_t_i_,axis=0)[:,None])
            self.Sigma_=np.array([np.matmul((X-self.mu_[k]).T,((X-self.mu_[k])*self.tau_t_i_[:,k].reshape(-1,1)))/np.exp(self.compute_vector_with_logs(self.tau_t_i_[:,k])) for k in range(self.k_)])            
            self.A_ = np.exp(self.compute_matrix_with_logs(self.tau_t_ij_,axis=0) - self.compute_matrix_with_logs(self.tau_t_ij_,axis=(0,1)))
           
            #E-Step
            Enp1=self.E_step(X,self.A_,self.pi_,self.pdf_k_s, self.tau_t_i_, self.tau_t_ij_)
            if (np.abs(Enp1/En-1)) < self.tol_:
                converged=True
            it+=1
            En=Enp1
            self.update_tau(X)
        
        # Uncomplete loglikelihood
        self.uncomplete_likelihood_ = En

        # Assigning labels
        self.labels_=self.predict_probable_sequence(X)
        
        # Computing complete likelihood
        self.complete_likelihood_ = self.compute_complete_likelihood(X,self.labels_)
     
    def predict_probable_sequence(self, X):
        ''' 
        Predict the most probable sequence for X
        
        Parameters:
        -----------
        X: (T, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment: np.array(T,)    
        '''              
        return self.viterbi(X, self.A_, self.pi_) 

########################################################################################### 
###########################################################################################
    
def plot_cov_elipse(ax,cov,mu,confidence=0.95,color="black"):
    """ 
    Plot the ellipse that contains a confidence
    percentage of the mass of the Gaussian distribution N(mu,cov)
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    
    confid_coef=chi2(df=2).ppf(confidence)
    theta = np.degrees(np.arctan2(*vecs.T[0][::-1]))    
    w, h = 2 * np.sqrt(confid_coef*vals)
    
    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=w, height=h,
                  angle=theta, color=color)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    
########################################################################################### 
###########################################################################################

def plot_results(X,X_test,model,train_label,test_label,model_name,color_plot="orangered",confidence=0.9,saving=False):
    """ 
    Plot the results of a model
    """
    colors=[color_plot]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.scatter(X.T[0],X.T[1], marker="2", c=train_label)
    ax1.scatter(model.mu_.T[0],model.mu_.T[1], c=colors, s=100, label="Centers")
    ax1.set_xlim(np.min(X)-0.5,np.max(X)+0.5)
    ax1.set_ylim(np.min(X)-0.5,np.max(X)+0.5)
    for sigma,mu in list(zip(model.Sigma_,model.mu_,)):
        plot_cov_elipse(ax1,sigma,mu,color="black",confidence=confidence)
    ax1.text(0.5,-0.11, "Loglikelihood : {}".format(np.round(model.compute_complete_likelihood(X, train_label),2)), size=12, ha="center",transform=ax1.transAxes, fontsize=16)
    ax1.set_title("{} on Train data - {}% confidence ellipses".format(model_name,int(confidence*100)), fontsize=13)
    ax1.legend()

    ax2.scatter(X_test.T[0],X_test.T[1], marker="2", c=test_label)
    ax2.scatter(model.mu_.T[0],model.mu_.T[1], c=colors, s=100, label="Centers")
    ax2.set_xlim(np.min(X)-0.5,np.max(X)+0.5)
    ax2.set_ylim(np.min(X)-0.5,np.max(X)+0.5)
    for sigma,mu in list(zip(model.Sigma_,model.mu_)):
        plot_cov_elipse(ax2,sigma,mu,color="black",confidence=confidence)
    ax2.text(0.5,-0.11, "Loglikelihood : {}".format(np.round(model.compute_complete_likelihood(X_test,test_label),2)), size=12, ha="center",transform=ax2.transAxes, fontsize=16)
    ax2.set_title("{} on Test data - {}% confidence ellipses".format(model_name,int(confidence*100)), fontsize=13)
    ax2.legend()
    
    if saving:
        plt.savefig(model_name+".png")

    plt.tight_layout()
    plt.show()