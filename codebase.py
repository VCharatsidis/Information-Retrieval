# %%

import numpy as np


# %%

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

# %%
def calculate_ERR(ranking):
    ERR = 0
    for r in range(len(ranking)):
        prob_to_stop_at_r = ranking[r]/(r+1)
        for i in range(r):
            prob_to_stop_at_r *= 1 - ranking[i]
            
        ERR += prob_to_stop_at_r
        
    return ERR
        
calculate_ERR(rankings[1][1])

# %%
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

# %%
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
  


# %%
def probabilistic_interleaving(list_a, list_b):
    interleaved_list = [ ]
    counter = 0
    
    while counter < 3:
        coin_toss = random.random()
                   
        if(coin_toss > 0.5):
            probs = softmax(list_a)
            chosen = np.random.choice(list_a, probs)
            list_a.remove(chosen)
            list_b.remove(chosen)
            counter += 1
        else:
            probs = softmax(list_b)
            chosen = np.random.choice(list_b, probs)
            list_b.remove(chosen)
            list_a.remove(chosen)
            counter += 1
           
            
    return interleaved_list
    
    
def softmax(rankings, tau = 3):
    numerators = 1 / np.power(rankings, tau)
    denominator = numerators.sum()
    
    return numerators / denominator
    

# %%
