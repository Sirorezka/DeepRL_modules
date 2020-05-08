import numpy as np

## BMAB - binary multiple arms bandit gittis index

class GittinsAgent():
    def __init__(self, L = 60, beta = 0.99):
        """
        L - lookahead window
        beta - discount factor
        
        calculation is based on following article:
            https://arxiv.org/pdf/1909.05075.pdf
        """
        self.beta = beta
        self.L = L
        self.get_idx = None
        
        self.ub = 1 ## upper bound for index
        self.lb = 0 ## lower bound for index
        self.eps = 0.001 ## epsilon for index calculations
    
    def init_actions(self, n_actions):
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0
        
    def get_v(self,p,q,lamb):

        L = self.L
        beta = self.beta

        v = np.zeros([L+1,L+1])

        p_ = np.arange(0,L+1)
        q_ = L-p_
        v[p_,q_] = beta**(L-1) / (1-beta) * np.maximum((p+p_) / (p+p_+q+q_)-lamb,0)


        for i in range(L-1,-1,-1):
            p_ = np.arange(0,i+1)
            q_ = i - p_
            v[p_,q_] = (p + p_) / (p + p_ + q + q_) - lamb + \
                       (p + p_) / (p + p_ + q + q_) * beta *  v[p_+1,q_] + \
                       (q + q_) / (p + p_ + q + q_) * beta *  v[p_,q_+1]
            v[p_,q_] = np.maximum(v[p_,q_],0)

        return v[0,0]


    def calc_gittins_index(self,p,q):
        ## https://arxiv.org/pdf/1909.05075.pdf
        up = self.ub
        lb = self.lb
        eps = self.eps
        while up-lb>eps:
            lambd = (up+lb) / 2
            v = self.get_v(p,q,lambd)
            if v>0:
                lb = lambd
            else:
                up = lambd

        return (up+lb) * 0.5

    def get_action(self):
        
        ## initial approximation
        ## push level that doesn't have failures
        best_idx = np.argmin(self._failures)
        if self._failures[best_idx] == 0:
            return best_idx
        

        best_action = np.argmax(self.get_idx)
        return best_action
          

    def update(self, action, reward):
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

        ## smoothing agents that doesn't have successes or failures    
        p = self._successes[action] + 1
        q = self._failures[action] + 1
        
        ## update gittins index 
        if self.get_idx is None:
            self.get_idx = np.zeros(len(self._successes))
        self.get_idx[action] = self.calc_gittins_index(p,q)
            
    @property
    def name(self):
        return self.__class__.__name__ + "(Gittins={})".format(self.beta)