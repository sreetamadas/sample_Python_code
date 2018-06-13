
# coding: utf-8

# In[7]:

## generate glycan profile for a given distribution of enzymes

## TO DO:
## use of function to 
##      1. randomly select the distribution of enzymes in a compartment
##      2. getting glycan profile for the selected enzyme distribution
## add extra condition to avoid beyond 2 branches (see if this is required ?)
## elegant way to print the tree

import numpy as np
#from anytree import Node, RenderTree
#from copy import deepcopy

# how to generate a graph or tree in python
# https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/
# https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
# https://stackoverflow.com/questions/3199171/append-multiple-values-for-one-key-in-python-dictionary


# In[8]:

'''
# function to create the branched glycans

class BranchedChain:
    def __init__(self):
        self.__rep = list()
        self.__char_set = dict()
        
    def ntips(self):
        return len(self.__rep)

    def get_chain(self, idx):
        assert idx < self.ntips()
        return self.__rep[idx]

    def size(self, charSet = None):
        if charSet is None:
            return np.sum( [self.__char_set[k]  for k in self.__char_set.keys()] )
        else:
            return self.__char_set[charSet] if charSet in self.__char_set else 0

    def extend_chain(self, charSet, idx = None):
        i = self.size(charSet)
        i += 1
        if idx is None:
            assert len(self.__rep) == 0
            assert len(self.__char_set) == 0
            self.__rep.append( [ (charSet, i) ] )
        else:
            assert idx < self.ntips()
            self.__rep[idx].append( (charSet, i) )
        self.__char_set[charSet] = i

    def create_branch(self, idx ):
        assert idx < self.ntips()
        # self.__rep.append( self.__rep[idx].copy() )
        new_list = list(self.__rep[idx])
        self.__rep.append(new_list)    
        return self.ntips()

    def get_tree(self):
        root = Node('root')
        visited, nodes = dict(),dict()
        nodes['root'] = root
        for lst in self.__rep:
            lastVisited = 'root'
            for charSet, idx in lst:
                name = '%s:%d' % (charSet, idx)
                if name not in visited:
                    visited[name] = True
                    nodes[name] = Node(name, parent=nodes[lastVisited])
                lastVisited = name
        return root

    def __str__(self):
        return str( self.__rep )
        #root = self.get_tree()
        #string = ''
        #for pre, fill, node in RenderTree(root):
        #    string = "%s\n%s%s" % (string,pre, node.name)
        #return string
        
'''


# In[ ]:

## step 1: generate all possible binary combinations of 8 digits (corresponding to 8 enzymes)
#https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value

'''
#method 1:
def per(n):
    for i in range(1<<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        print map(int,list(s))

num_enzymes = 8
per(num_enzymes)
'''


# In[39]:

# method 2:
import itertools
num_enzymes = 7 #8
# since the first enzyme should be present in the 1st compartment & not in later compartments,
# we can leave this out for now
lst = map(list, itertools.product([0, 1], repeat = num_enzymes))

lst
#master_list[1]


# In[ ]:

#### mapping file for all enzyme reactions ####

#P/0 -> A/1 (yellow sq) -> B/2 (blue sq)   -> C/3 (red tri)
#  1                    2                3

#p ->   A/1 (yellow sq) -> D/4 (yellow cr) -> B/2 (blue sq)
#                       4                5
                      
#------------------------  D/4 (yellow cr) -> E/5 (brown diamond)
#                                       6
                                     
#------------------------  B/2 (blue sq)   -> D/4 (yellow cr)
#                                       7

#-----  A/1 (yellow sq) -> E/5 (brown rhombus)
#                       8


# In[6]:

# these two have not been used as of now
# create a dictionary with sugars that support branching
branched_sugar = {'1':'' , '2':'' , '4':''}

# create a dictionary of capping sugars
cap = {'5':''}


# In[3]:

# create a 2D array with the sequence: enzyme no, 1st substrate, 2nd substrate
map = []
# map.append([1,0,1])   # leave out the first enzyme
map.append([2,1,2])
map.append([4,1,4])
map.append([8,1,5])
map.append([3,2,3])
map.append([7,2,4]) # these 2 enzymes being present in the same compartment will give rise to cyclisation
map.append([5,4,2]) # eliminate correponding combinations from the master_list at the very beginning ?
map.append([6,4,5])

map
#map[1] = [4, 1, 4]
#map[1][1] = 1  # map[i][j] -> ith row (group), jth col (element in the group)


# In[6]:

'''
# initialize 2 branches with the 1st sugar
branch1 = []
branch1.append(1)
branch2 = []
branch2.append(1)
'''


# In[11]:

'''
# initialize the sugar using class
molecule = BranchedChain()
molecule.extend_chain('1')
print(molecule)
'''


# In[56]:

# intialise the sugar as a dictionary

mol = {'1_1':''}  # sugarname_position; the keys are in string format


# In[ ]:

##################################################################################################


# In[5]:

# select an entry from master list at random - gives distribution of enzymes in the compartment

#### compartment 1 ######
import random
#comp_distr = random.choice(lst)
#comp_distr


# In[ ]:

comp_distr = [0, 0, 1, 0, 0, 1, 1]


# In[58]:

# foreach item in the above list (of enzyme distribution), if value=1, select the corresponding reaction index from map
# then check if existing glycan list already contains sugars to which the new ones can be appended;
# allow branching where possible : add extra condition to avoid beyond 2 branches

cnt = 2
for i in range(num_enzymes): #since, len(comp_distr) = num_enzymes
    if comp_distr[i]==1:
        print('')
        print(map[i])
        # check if the left element of the selected reaction is already present in the glycan chain
        for k in range(1,cnt):  # this indicates values from 1 to cnt-1
            #print(k)
            #print(str(map[i][1])+'_'+str(k))
            if (str(map[i][1])+'_'+str(k)) in mol:
                mol[(str(map[i][1])+'_'+str(k))] = (mol[(str(map[i][1])+'_'+str(k))] + ', '+ str(map[i][2]) + '_' + str(k+1))  
                mol[(str(map[i][2])+'_'+str(k+1))] = ''
                cnt = cnt+1


# In[59]:

mol


# In[ ]:

##################################################################################################


# In[40]:

#### compartment 2 ######
#comp_distr = random.choice(lst)
#comp_distr


# In[60]:

comp_distr = [1, 1, 0, 0, 0, 1, 0]


# In[61]:

#cnt = 2+1
for i in range(num_enzymes): #since, len(comp_distr) = num_enzymes
    if comp_distr[i]==1:
        print('')
        print(map[i])
        # check if the left element of the selected reaction is already present in the glycan chain
        for k in range(1,cnt):  # this indicates values from 1 to cnt-1
            #print(k)
            #print(str(map[i][1])+'_'+str(k))
            if (str(map[i][1])+'_'+str(k)) in mol:
                #mol[(str(map[i][1])+'_'+str(k))].append(str(map[i][2])+'_'+str(k+1))
                mol[(str(map[i][1])+'_'+str(k))] = (mol[(str(map[i][1])+'_'+str(k))] + ', '+ str(map[i][2]) + '_' + str(k+1)) 
                mol[(str(map[i][2])+'_'+str(k+1))] = ''
                cnt = cnt+1


# In[53]:

#mol


# In[33]:

#range(1,3)


# In[62]:

for key in mol:
    print(key + '->' + mol[key])


# In[ ]:

##################################################################################################


# In[63]:

#### compartment 3 ######
#comp_distr = random.choice(lst)
#comp_distr


# In[ ]:

comp_distr = [1, 1, 0, 1, 1, 0, 0]


# In[64]:

for i in range(num_enzymes): #since, len(comp_distr) = num_enzymes
    if comp_distr[i]==1:
        print('')
        print(map[i])
        # check if the left element of the selected reaction is already present in the glycan chain
        for k in range(1,cnt):  # this indicates values from 1 to cnt-1
            #print(k)
            #print(str(map[i][1])+'_'+str(k))
            if (str(map[i][1])+'_'+str(k)) in mol:
                #mol[(str(map[i][1])+'_'+str(k))].append(str(map[i][2])+'_'+str(k+1))
                mol[(str(map[i][1])+'_'+str(k))] = (mol[(str(map[i][1])+'_'+str(k))] + ', '+ str(map[i][2]) + '_' + str(k+1)) 
                mol[(str(map[i][2])+'_'+str(k+1))] = ''
                cnt = cnt+1


# In[65]:

for key in mol:
    print(key + '->' + mol[key])


# In[ ]:



