## STEP 1: generate all possible binary combinations of 7 digits (corresponding to the enzymes present/absent in a compartment)

#https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
'''
#method 2:
def per(n):
    for i in range(1<<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        print map(int,list(s))

num_enzymes = 8
per(num_enzymes)
'''

# method 2:
import itertools
# since the first enzyme should be present in the 1st compartment & not in later compartments,
# we can leave this out for now
num_enzymes = 7   # 8
lst = map(list, itertools.product([0, 1], repeat=num_enzymes))
lst
#lst[1]

###################################################################################################
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
#                    8
###################################################################################################

# STEP 2: create a 2D array with the sequence: enzyme no, 1st substrate, 2nd substrate - this is mapping file for the enzyme reactions

map = []
# map.append([1,0,1])   # leave out the first enzyme
map.append([2,1,2])
map.append([4,1,4])
map.append([8,1,5])
map.append([3,2,3])
map.append([7,2,4])
map.append([5,4,2])
map.append([6,4,5])
# map   # map[1] ;  #map[1][1]  # map[i][j] -> ith row, jth col


# STEP 3: # initialize 2 branches with the 1st sugar
branch1 = []
branch1.append(1)
branch2 = []
branch2.append(1)


# STEP 4: # select an entry from lst at random - gives distribution of enzymes in the compartment
import random
comp_distr = random.choice(lst)
#comp_distr


# STEP 5: create the glycan chains for the above assignment in a compartment
# foreach item in the above list, if value=1, select the corresponding index from map

for i in range(num_enzymes): #since, len(comp_distr) = num_enzymes
    if comp_distr[i]==1:
        print map[i]
        

