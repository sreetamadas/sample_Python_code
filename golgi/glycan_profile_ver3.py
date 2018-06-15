
# coding: utf-8

# In[ ]:

## generate glycan profile for a given distribution of enzymes

## TO DO:
## use of function to 
##      1. randomly select the distribution of enzymes in a compartment
##      2. getting glycan profile for the selected enzyme distribution
## add extra condition to avoid beyond 2 branches (see if this is required ?)
## elegant way to print the tree
## also, this generates only 1 product - modify to generate all the products


# how to generate a graph or tree in python
# https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/

# https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
# https://stackoverflow.com/questions/3199171/append-multiple-values-for-one-key-in-python-dictionary

# find all possible subgraphs of a graph in python
# find all sub-graphs starting from root node in a directed tree/graph in python


# In[1]:

## step 1: generate all possible binary combinations of 8 digits (corresponding to 8 enzymes)
#https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value

import itertools
num_enzymes = 7 #8
# since the first enzyme should be present in the 1st compartment & not in later compartments,
# we can leave this out for now

lst = map(list, itertools.product([0, 1], repeat = num_enzymes))
#lst


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


# In[2]:

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


# In[3]:

# intialise the sugar as a dictionary
mol = {'1_1':''}  # sugarname_position; the keys are in string format


# In[4]:

# foreach item in the above list (of enzyme distribution), if value=1, select the corresponding reaction index from map
# then check if existing glycan list already contains sugars to which the new ones can be appended;
# allow branching where possible : add extra condition to avoid beyond 2 branches

def get_gly(cnt,comp_distr,mol):   # cnt,num_enzymes,map
    
    # select enzyme distribution
    #comp_distr = random.choice(lst)
    #comp_distr = [0, 0, 1, 0, 0, 1, 1]   ## comp 1
    #comp_distr = [1, 1, 0, 0, 0, 1, 0]   ## comp 2
    #comp_distr = [1, 1, 0, 1, 1, 0, 0]    ## comp 3
    #print(comp_distr)
    
    # get glycan
    for i in range(num_enzymes): #since, len(comp_distr) = num_enzymes
        if comp_distr[i]==1:
            #print('')    #print(map[i])
            
            # check if the left element of the selected reaction is already present in the glycan chain
            for k in range(1,cnt):  # this indicates values from 1 to cnt-1
                #print(k)   #print(str(map[i][1])+'_'+str(k))
                if (str(map[i][1])+'_'+str(k)) in mol:
                    mol[(str(map[i][1])+'_'+str(k))] = (mol[(str(map[i][1])+'_'+str(k))] + ', '+ str(map[i][2]) + '_' + str(k+1))  
                    #mol[(str(map[i][2])+'_'+str(k+1))] = ''
                    if(str(map[i][2])+'_'+str(k+1)) in mol:
                        mol[str(map[i][2])+'_'+str(k+1)] = mol[str(map[i][2])+'_'+str(k+1)]
                    else:
                        mol[(str(map[i][2])+'_'+str(k+1))] = ''
                    cnt = cnt+1
                
    return (mol,cnt)


# In[ ]:

#######################################  compartment 1  #########################################################


# In[5]:

#import random
#comp_distr = random.choice(lst)
comp_distr = [0, 0, 1, 0, 0, 1, 1]

mol,Cnt = get_gly(2,comp_distr,mol)     # 2,num_enzymes,map


# In[6]:

for key in mol:
    print(key + '->' + mol[key])


# In[7]:

################# compartment 2 #####################
#comp_distr = random.choice(lst)
comp_distr = [1, 1, 0, 0, 0, 1, 0] 

mol,Cnt = get_gly(Cnt,comp_distr,mol)


# In[8]:

for key in mol:
    print(key + '->' + mol[key])


# In[9]:

################# compartment 3 #####################
#comp_distr = random.choice(lst)
comp_distr = [1, 1, 0, 1, 1, 0, 0]

mol,Cnt = get_gly(Cnt,comp_distr,mol)


# In[10]:

for key in mol:
    print(key + '->' + mol[key])


# In[ ]:

##############################################################################################
## generate the product profile at the end
# STEPS: 
# clean-up : remove keys with no value assigned
# consider sub-graphs which include only the key 1_1 (root node), but not those with contain only 2_2 or 2_3
#                                    (since these cant start assembly)
# consider all valid combinations


# or, generate the tree using this link : https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/
# so that clean-up is not required


# In[11]:

# clean-up : remove keys with no value assigned
clean_mol = dict([(vkey, vdata) for vkey, vdata in mol.iteritems() if(str(vdata).strip()) ])

for key in clean_mol:
    print(key + '->' + clean_mol[key])


# In[12]:

# clean-up : remove ', ' at the beginning, split by ', ', & then remove repetitions in the values
for key in clean_mol:
    clean_mol[key] =  ' '.join(list(set(clean_mol[key][2:].split(', ')))) # list(set(mol['1_1'][2:].split(', ')))

for key in clean_mol:
    print(key + '-> ' + clean_mol[key])


# In[ ]:

# print valid sub-graphs starting from root note 1_1

