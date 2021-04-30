######################################################
#  ('aabbcc', [1, 2, 1, 2, 1, 2])
# how to remove adjacent repeats with minimal cost, such that string has no repeats

def solution(S, C):
    # write your code in Python 3.6

    # split the string; need to keep stock of the indices
    tmp_arr = list(S)
    #print('arr',tmp_arr)

    # save indices of repeats
    ind = []
    # since we are comparing, final index is less than len(array)
    for i in range(0,len(tmp_arr)-1):
        if tmp_arr[i] == tmp_arr[i+1]:
            if C[i] <= C[i+1]:
                ind.append(i)
            else:
                ind.append(i+1)
    #print(ind)

    # get cost
    cost = 0
    for i in ind:
        cost = cost + C[i] #C[ind[i]]
    #print('cost',cost)
    return cost
    #pass

##########################################################
# find how many empty plots whih satisfy the condition: empty plot is < k units from each house
# A denotes the matrix of plots - 0 is empty, 1 has house
#K=2, A = A = [[0,0,0,0],[0,0,1,0],[1,0,0,1]]

def solution(K, A):
    # write your code in Python 3.6

    # extract empty & house plots
    empty = []
    house = []
    for i in range(0,len(A)):
        for j in range(0,len(A[i])):
            if A[i][j] == 0:
                empty.append((i,j))
            elif A[i][j] == 1:
                house.append((i,j))
    #print(empty)

    # calc dist, & save entries with dist < K
    plot = []
    for empty_x,empty_y in empty:
        flag = 0
        for house_x,house_y in house:
            dist = abs(house_x - empty_x) + abs(house_y - empty_y)
            if dist > K: #<= K*len(house):
                flag = flag + 1
        if flag == 0:
            #distance.append(dist)
            plot.append((empty_x,empty_y))

    # return no. of plots
    #print(plot)
    #print('no', len(plot))
    return len(plot)
    #pass

######################################################
# array denotes pollution from factories, each filter on a factory halves pollution. find min no of filters to halve total pollution
#A = [5,19,8,1]

def solution(A):
    # write your code in Python 3.6

    # get target pollution limit
    target = sum(A)/2

    # logic: always reverse sort, and start with halving the highest polluter
    sorted_A = sorted(A, reverse = True)

    #for i in range(len(sorted))
    filters = 0
    while sum(sorted_A) > target:
        #for i in range(len(sorted)):
        sorted_A[0] = sorted_A[0]/2
        sorted_A = sorted(sorted_A, reverse=True)
        filters = filters + 1
    #print('no. of filters', filters)
    return filters
    pass

#############################################################
# find the smallest positive integer which is missing in the list A
A = [1, 3, 6, 4, 1, 2]

def cmp(A):
    if max(sorted(A)) <= 0:
        req = 1
    else:
        cmp_lst = list(range(1,max(sorted(A))+1))
        #req = 10000001
        req_list = []
        for i in cmp_lst:
            if i not in sorted(A): #and i < req:
                req_list.append(i)# = i
                #break;
            #else:
            #    req = i+1
        if len(req_list) == 0:
            req = max(sorted(A))+1
        else:
            req = min(req_list)
    return(req)


A = [1, 3, 6, 4, 1, 2]  
cmp(A)

A = [1,2,3]
cmp(A)

A = [1]
cmp(A)

A = [-1, 0]
cmp(A)
