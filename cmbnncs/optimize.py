import numpy as np
import math
import matplotlib.pyplot as plt

class LearningRateDecay:
    def __init__(self, iteration, iteration_max=10000, lr=0.1, lr_min=1e-6):
        ''' learning rate decays with iteration. '''
        self.lr = lr
        self.lr_min = lr_min
        self.iteration = iteration
        self.iteration_max = iteration_max

    def exp(self, gamma=0.999, auto_params=True):
        '''
        exponential decay, return: lr * gamma^iteration
        
        :param auto_params: if True, gamma is set automatically        
        '''
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./self.iteration_max)
        return self.lr * gamma**self.iteration
    
    def step(self, stepsize=1000, gamma=0.3, auto_params=True):
        '''
        learning rate decays step by step, similar to 'exp'
        
        :param auto_params: if True, gamma is set automatically        
        '''
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./(self.iteration_max/stepsize))
        return self.lr * gamma**(math.floor(self.iteration/stepsize))
    
    def poly(self, decay_step=500, power=0.999, cycle=True):
        ''' polynomial decay, return: (lr-lr_min) * (1 - iteration/decay_steps)^power +lr_min '''

        if cycle:
            d_steps = decay_step * max(1, math.ceil(self.iteration/decay_step))
        else:
            d_steps = self.iteration_max
        # factor asegura que no sea negativo si la iteración supera d_steps
        factor = max(0, 1 - self.iteration/d_steps)
        return (self.lr - self.lr_min) * (factor**power) + self.lr_min

# --- TEST ---
if __name__=='__main__':
    total_iterations = 10000
    lr_all_1 = []
    for ite in range(1, total_iterations + 1):
        lrd = LearningRateDecay(ite, iteration_max=total_iterations, lr=7.4e-5)
        lr_all_1.append(lrd.exp())
    plt.figure(figsize=(8,6))
    plt.plot(lr_all_1, color='tab:blue', linewidth=2, label='Exponencial (Recomendado)')
    plt.title('Estrategia de Learning Rate para Limpieza de CMB', fontsize=14)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.show()