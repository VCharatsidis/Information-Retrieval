
# coding: utf-8

# In[1]:



import numpy as np
import random


# In[2]:



def simulate_rankings():
    P = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0]]
    E = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0]]
    rankings = []
    for p in P:
        for e in E:
            tup1 = (p, e)
            rankings.append(tup1)
    
    print(rankings)
    return rankings
    
rankings = simulate_rankings()


# In[3]:


def calculate_ERR(ranking):
    ERR = 0
    for r in range(len(ranking)):
        prob_to_stop_at_r = ranking[r]/(r+1)
        for i in range(r):
            prob_to_stop_at_r *= 1 - ranking[i]
            
        ERR += prob_to_stop_at_r
        
    return ERR
        
calculate_ERR(rankings[1][1])


# In[4]:


def calculate_Dmeasures(rankings):
    
    measures = {k:[] for k in range(10)}
    for index, r in enumerate(rankings):
        ERR_P = calculate_ERR(r[0])
        ERR_E = calculate_ERR(r[1])
        
        d_measure = ERR_E - ERR_P
        
        if d_measure >= 0.05 and d_measure <= 0.95:
            measures[int(d_measure * 10)].append(index)
            
    
    return measures

measures = calculate_Dmeasures(rankings)
print(measures)


# In[5]:


def team_draft_interleaving(list_a, list_b):
    
    interleaved_list = [ ]
    counter = 0
    while counter < 3:
        coin_toss = random.random()
                   
        if(coin_toss > 0.5):
            put_first_available_url_in_interleaved(list_a, 0, interleaved_list)
            counter += 1
            if(counter == 2):
                return interleaved_list
            
            put_first_available_url_in_interleaved(list_b, 1, interleaved_list)  
            counter += 1
        else:
            put_first_available_url_in_interleaved(list_b, 1, interleaved_list)
            counter += 1
            if(counter == 2):
                return interleaved_list
            
            put_first_available_url_in_interleaved(list_a, 0, interleaved_list)
            counter += 1
            
    return interleaved_list

    
def put_first_available_url_in_interleaved(a_list, index_list, interleaved_list):
    for i in a_list:
        if i not in interleaved_list:
            tup = (i, index_list)
            intearleaved_list.append(tup)
            return
  


# In[68]:


def probabilistic_interleaving(list_a, list_b):
    interleaved_list = [ ]
    counter = 0
    
    while counter < 3:
        coin_toss = random.random()
                   
        if(coin_toss > 0.5):
            probs = softmax(list_a)
            chosen = np.random.choice(list_a, 1, p = probs)
            tup = (int(chosen), 0)
            interleaved_list.append(tup)
               
        else:
            probs = softmax(list_b)
            chosen = np.random.choice(list_b, 1, p = probs)
            tup = (int(chosen), 1)
            interleaved_list.append(tup)
      
        counter += 1  
        if chosen in list_a: 
            list_a.remove(chosen)
            
        if chosen in list_b: 
            list_b.remove(chosen)
            
    return interleaved_list
    
    
def softmax(a_list, tau = 3):
    rankings = []
    for i,doc in enumerate(a_list):
        rankings.append(i+1)
    
    numerators = 1 / np.power(rankings, tau)
 
    denominator = numerators.sum()
   
    return numerators / denominator
    


# In[73]:


list_a = [3, 1, 5]
list_b = [1, 2, 10]

probs = softmax(list_a)
print(probs)
res = np.random.choice(list_a, 1, p = probs)
print(res)
interleaved = probabilistic_interleaving(list_a, list_b)
print(interleaved)

