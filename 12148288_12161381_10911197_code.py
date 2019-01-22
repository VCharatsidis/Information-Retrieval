# %%
import numpy as np

# %%
'''
### Step 1: Simulate Rankings of Relevance for E and P
'''

# %%
def simulate_rankings():
    ''' This method creates all possible combinations of rankings for P and E.
    
        @Output: a list of tuples. A tuple has 2 integer arrays (one ranking for P and one for E).
    '''
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
'''
$\textbf{Step 2:}$ Calculate the $\Delta$ measure

We calculate the Expected reciprocal rank of a ranking using the guy 

ERR := $\sum_{r = 1}^n \frac{1}{r}$ P(user stops at position r)

where n is the number of documents in the ranking and

$P = \prod_{i=1}^{r-1} (1-R_i)R_r$

where $R_i = \frac{2^g - 1}{2^{g_{max}}}$ where $g_i$ is the grade of the i-th document and $g_{max}$ is the the maximum relevance.

Putting it all together ERR:= $\sum_{r = 1}^n \frac{1}{r} \prod_{i=1}^{r-1} (1-R_i)R_r$ 

As defined in the paper Expected Reciprocal Rank for Graded Relevance. Olivier Chapelle, Ya Zhang, Donald Meltzler, Pierre Grinspan.
'''

# %%
def calculate_ERR(ranking):
    ''' This method calculates the ERR of a ranking. A ranking is a one dimensional interger list with length 3.
    
        @Input: a one dimensional list of length 3 with zeros and ones.
        
        @Output: a double (ERR score).
    '''
    ERR = 0
    for r in range(len(ranking)):
        prob_to_stop_at_r = ranking[r]/(r+1)
        for i in range(r):
            prob_to_stop_at_r *= 1 - ranking[i]
            
        ERR += prob_to_stop_at_r
        
    return ERR

calculate_ERR(rankings[5][1])
# rankings[5][1]

# %%
'''
Then we calculate the $\Delta $measures and split them in 10 buckets.

The buckets are made such that group 1 contains all pairs for which 0.05 < $\Delta$ measure ≤ 0.1, group 2 all pairs for which 0.1 < $\Delta$measure ≤ 0.2, etc.
'''

# %%
def calculate_Dmeasures(rankings):
    ''' This method calculates the difference in ERR between the two rankings of every tuple, 
        for all tuples of rankings.
        Since we need to store the results in 10 different buckets we use a dictionary with keys 10 integers 
        from 0 to 9 with step 1 and values the list of our measurements.
    
        @Input: a list of tuples.
        
        @Output: A dictionary with keys integers from 0 to 9 and values 10 lists of doubles. 
        
    '''
    measures = {k:[] for k in range(10)}
    
    for index, r in enumerate(rankings):
        ERR_P = calculate_ERR(r[0])
        ERR_E = calculate_ERR(r[1])
        
        d_measure = ERR_E - ERR_P
        
        if d_measure >= 0.05 and d_measure <= 0.95:
            measures[int(d_measure * 10)].append(index)
            
    return measures

# %%
'''
$\textbf{Step 3}$: Implement Team-Draft Interleaving and Probabilistic Intearleaving

$\textbf{Team Draft Interleaving}$ is performed by throwing a coin. If its heads we put, in the interleaved list,
the first document of list A that is not already in and then the first document of list B that is not already in,
if its tails we start with B and then A.
We repeat until the interleaved list if full.

        
'''

# %%
def team_draft_interleaving(list_a, list_b):
    ''' Team draft interleaving is performed by throwing a coin. If its heads we put in the interleaved list
        the first element of list A that is not already in,
        if its tails we put in the interleaved list the first element of list B that is not already in.
        
        @Input: 2 lists of intergers of length 3.
        
        @Output: an interleaved list of tuples of length 3. Each tuple contain an integer (document ID) and a 0 or 1 
        depending of which list it came from.
    
    '''
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
    ''' Helper method that creates a tuple with and integer (document ID) and 
        a 0 or 1 which indicates from which list it came from and adds it in the interleaved list. 
        
        @Input: - a list of intergers of length 3.
                - a 0 or 1 indicator of the list.
                - the interleaved list with all tuples of rnakings.
         
    '''
    for i in a_list:
        already_in_interleaved = False
        for tupl in interleaved_list:
            if tupl[0] == i:
                already_in_interleaved = True
                
        if i not in already_in_interleaved:
            tup = (i, index_list)
            intearleaved_list.append(tup)
            return
  


# %%
'''
$\textbf{Probabilistic Intearleaving}$ is performed similarly but instead of choosing the first document we chose the documents from the lists probabilisticly where the probabilities of the documents are produced from 2 softmax functions
(one for every list).

The softmax function assigns probability of selecting a document ($P_{s_{x}}(d)$) that is inversely proportional to a power of the rank $r_x(d)$ of a document d in a list.

$P_{s_{x}}(d) = \frac{ \frac{1}{r_x(d)^\tau}}{\sum_{d2\in D} \frac{1}{r_x(d2)^\tau}}$ 

where the parameter $\tau$ controls how quickly selection probabilities decay as rank decreases.

as defined in the paper A Probabilistic Method for Inferring Preferences from Clicks. Katja Hofmann, Shimon Whiteson and Maarten de Rijke. University of Amsterdam
'''

# %%

def probabilistic_interleaving(list_a, list_b):
    ''' Probabilistic interleaving perfromed by throwing coins, if its heads we use the softmax function of list A
        to choose randomly (with higher probability the more relevant rankings)
        else we use the softmax function of list B. 
        After we put the choosen doc we remove it from both lists A and B softmaxes.
        We do this until the interleaved list is full.
        The interleaved list contains tuples with the document ID and a 0 or 1 considering the list that it came from.
        
        @Input: 2 lists of intergers of length 3.
        
        @Output: an interleaved list of tuples of length 3. Each tuple contain an integer (document ID) and a 0 or 1 
        depending of which list it came from.
    
    '''
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
    ''' Helper method that calculates the probabilities of every document in the given list
        using the softmax function in a vectorised from.
        
        @Input: list of intergers of length 3 (rankings).
        
        @Output: a vector with probabilities for every document.
    '''
    numerators = 1 / np.power(rankings, tau)
    denominator = numerators.sum()
    
    return numerators / denominator
    

# %%
'''
### Part 5: Simulate Interleaving Experiment
'''

# %%
def run_interleave_experiments(k):
    for exp in interleave_experiments:
        results = []a
        for i in range(k):
            click_results_E, click_results_P = get_click_results()
            
            if click_results_E > click_results_P:
                results.append(click_results_E)
                
#                 ? What do to with these res
    return