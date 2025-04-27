import numpy as np
import math
import random
from pqdict import pqdict
import copy
import time
import pickle as pickle
import argparse

#TODO: do we want self-renewing resources?

#input arguments
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--g', default=1e4, type=float)
parser.add_argument('-k', '--kappa', default=1e8, type=float)
parser.add_argument('-z1', '--Z1', default=0.01, type=float)
parser.add_argument('-z2', '--Z2', default=1, type=float)
parser.add_argument('-ga', '--gamma', default=1, type=float)
parser.add_argument('-p', '--mut_prob', default=1, type=float)
parser.add_argument('-l_phi', '--lphi', default=1, type=float)
parser.add_argument('-l_psi', '--lpsi', default=1, type=float)
parser.add_argument('-l_evo', '--levo', default=20, type=float)
parser.add_argument('-n', '--num_reactions', default=10000000, type=float)
parser.add_argument('-L', '--num_sites', default=300, type=float)
parser.add_argument('-o', '--outfile', default='replicative_homogeneous.p', type=str)
args = parser.parse_args()

# periodic boundary conditions
def compute_distance(c1, c2, L):
    dc = np.abs(c1 - c2)
    if dc > L/2:
        dc = L - dc
    return dc

# exponential kernels
def kernel(c1, c2, L, decay_length, Z):
    # normalized
    return np.exp(-compute_distance(c1, c2, L)/decay_length)/Z

# pre-compute normalization
def normalization(L, decay_length):
    return sum([kernel(0, i, L, decay_length, 1) for i in range(L)])

# declare different reactions
# declare different reactions
class ResourceBirth(object):
    def __init__(self, mother, rate, L):
        self.mother = mother
        self.stoich = {self.mother: 1}
        self.rate_const = rate
        self.reactants = [self.mother]
        self.type='resourcebirth'
    
    def offspring(self):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]
        
class ResourceDeath(object):
    def __init__(self, mother, rate, L):
        self.mother = mother
        self.stoich = {self.mother: -1}
        self.rate_const = rate
        self.reactants = [self.mother]
        self.type='resourcedeath'
        
    def offspring(self):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]*state[self.mother]
    
class ConsumerDeath(object):
    def __init__(self, mother, rate, L):
        self.mother = mother
        self.stoich = {self.mother: -1}
        self.rate_const = rate
        self.reactants = [self.mother]
        self.type='consumerdeath'
        
    def offspring(self):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]

class Consumption(object):
    def __init__(self, mother, food, rate, L, l_phi, Z):
        self.mother = mother
        self.food = food
        self.stoich = {self.food: -1}
        self.rate_const = rate*kernel(mother-L, food, L, l_phi, Z)
        self.reactants = [self.mother, self.food]
        self.type='consumption'
        
    def offspring(self):
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]*state[self.food]
    
class ConsumptionBirth(object):
    def __init__(self, mother, food, rate, L, l_psi, Z, p, levo):
        self.mother = mother
        self.daughter = mother
        self.mutants = list((np.arange(mother-levo, mother+levo+1).astype(int)-L)%L+L)
        self.mut_prob = p
        self.food = food
        self.stoich = {self.food: -1, self.daughter: 1}
        self.rate_const = rate*kernel(mother-L, food, L, l_psi, Z)
        self.reactants = [self.mother, self.food]
        self.type='consumptionbirth'
    
    def offspring(self):
        if random.random() < self.mut_prob:
            self.daughter = random.choice(self.mutants)
            self.stoich = {self.food: -1, self.daughter: 1}
        else:
            self.daughter = self.mother
            self.stoich = {self.food: -1, self.daughter: 1}
        return

    def propensity(self, state):
        return self.rate_const*state[self.mother]*state[self.food]
       
#### Main loop ####

def main_simulation(reactions, inv_map, x, num_events):
# Generate random numbers, times, priority queue
    t=0
    T=[t]
    X=[copy.deepcopy(x)]
    schedule = pqdict()
    record = max(num_events/10000, 1)
    
    for i in range(len(reactions)):
        a = reactions[i].propensity(x)
        if a==0:
            schedule[i] = math.inf
        else:
            schedule[i] = -math.log(random.random())/a
        
    
    rnext, tnext = schedule.topitem()
    t = tnext
    dep_set = set()
    reactions[rnext].offspring()
    for k, v in reactions[rnext].stoich.items():
        x[k]=x[k]+v
        dep_set.update(inv_map[k])
    n=1

    while n < num_events:
        # reschedule reaction
        a = reactions[rnext].propensity(x)
        if a==0:
            schedule[rnext] = math.inf
        else:
            schedule[rnext] = t-math.log(random.random())/a

        # reschedule dependent reactions
        for r in dep_set:
            a = reactions[r].propensity(x)
            if a==0:
                schedule[r] = math.inf
            else:
                schedule[r] = t-math.log(random.random())/a
        
        # next reaction
        rnext, tnext = schedule.topitem()
        t = tnext
        dep_set = set()
        reactions[rnext].offspring()
        for k, v in reactions[rnext].stoich.items():
            x[k]=x[k]+v
            dep_set.update(inv_map[k])
        n+=1
        # choose when to record data
        if n%record==0:
            if sum(x[:int(len(x)/2)])<=0:
                break
            else:
                #print(n)
                T.append(t)
                X.append(copy.deepcopy(x))

    return T, X

##### Initialize #####
print('initializing...', flush=True)

# number of sites
L = args.num_sites

# print fixed point values
cstar = int(args.g*(1-args.gamma/(args.Z1*args.kappa))/(args.Z1+args.Z2))
rstar = int(args.gamma/args.Z1)

print('consumer fixed point value = '+str(cstar))
print('resource fixed point value = '+str(rstar))
print('Effective diffusion constant = '+str(args.mut_prob/2))
print('Relaxation parameter = ' +str((args.kappa*args.Z1/args.gamma-1)*args.lpsi**2))


##### Create reaction dictionaries #####

reactions = {}

# Resource births
rate = args.g
for i in range(L):
    reactions[i] = ResourceBirth(i, rate, L)
print('created resource birth reactions...', flush=True)
    
# Resource deaths
rate = args.g/args.kappa
for i in range(L):
    reactions[L+i] = ResourceDeath(i, rate, L)
print('created resource death reactions...', flush=True)

# Consumer deaths
rate = args.gamma
for i in range(L):
    reactions[2*L+i] = ConsumerDeath(L+i, rate, L)
print('created consumer death reactions...', flush=True)

# Consumption
temp = 0
rate = args.Z2
l_phi = args.lphi
Z = normalization(L, l_phi)
for i in range(L):
    for j in range(L):
        reactions[3*L+temp] = Consumption(L+i, j, rate, L, l_phi, Z)
        temp+=1
print('created consumption reactions...', flush=True)

# Consumption+Birth
temp = 0
rate = args.Z1
l_psi = args.lpsi
Z = normalization(L, l_psi)
for i in range(L):
    for j in range(L):
        reactions[3*L+L**2+temp] = ConsumptionBirth(L+i, j, rate, L, l_psi, Z, args.mut_prob, args.levo)
        temp+=1
print('created consumption+birth reactions...Done creating reactions!', flush=True)


##### Create reaction dependency graph #####
react_dict = {}

for k,v in reactions.items():
    react_dict[k] = v.reactants

inv_map = {}
for k, v in react_dict.items():
    for i in v:
        inv_map.setdefault(i,set()).add(k)

print('created dependency graph!', flush=True)

##### Initialize #####
# resources are first L indices, consumers are next L indices
x = cstar*np.ones(2*L)
for i in range(L):
    x[i]=rstar

# number of interactions
num_events = args.num_reactions

# time the simulation
t1 = time.time()

print('Number of reactions executed:', flush=True)
# run simulation
T,X = main_simulation(reactions, inv_map, x, num_events)

# print simulation time
print('Total time: '+str(time.time()-t1), flush=True)

res = []
cons = []
for i in X:
    res.append(i[:L])
    cons.append(i[L:])

# save output
pickle.dump([T, res, cons, args], open( args.outfile, "wb" ) )
