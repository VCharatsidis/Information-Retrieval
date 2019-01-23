# %%
# Import all dependencies
import numpy as np
import random as rd
import pandas as pd
import math
import re
from scipy.stats import norm
from copy import deepcopy

# %%
'''
### Step 1: Simulate Rankings of Relevance for E and P
'''

# %%


def simulate_rankings():
    ''' This method creates all possible combinations of rankings for P and E.

        @Output: a list of tuples. A tuple has 2 integer arrays (one ranking for P and one for E).
    '''
    P = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
         [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0]]
    E = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
         [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0]]
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
### Step 2: Calculate the $\Delta$ measure

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
    measures = {k: [] for k in range(10)}

    # The indices in measures are now the indices of the pair tuples
    for index, r in enumerate(rankings):
        ERR_P = calculate_ERR(r[0])
        ERR_E = calculate_ERR(r[1])

        d_measure = ERR_E - ERR_P

        if d_measure >= 0.05 and d_measure <= 0.95:
            measures[int(d_measure * 10)].append(index)

    return measures


# %%
'''
### Step 3: Implement Team-Draft Interleaving and Probabilistic Intearleaving

$\textbf{Team Draft Interleaving}$ is performed by throwing a coin. If its heads we put, in the interleaved list,
the first document of list A that is not already in and then the first document of list B that is not already in,
if its tails we start with B and then A.
We repeat until the interleaved list if full.

        
'''

# %%
# from IPython.core.debugger import set_trace


def convert_lists_to_labeled(list_a, list_b):
    considered_lists = [deepcopy(list_a), deepcopy(list_b)]
    label_results = []
    # set_trace()

    for doc_list in considered_lists:
        possible_labels = rd.sample(range(1, 20), 3)
        label_results.append(list(zip(doc_list, possible_labels)))

    return label_results


def team_draft_interleaving(list_a, list_b):
    ''' Team draft interleaving is performed by throwing a coin. If its heads we put in the interleaved list
        the first element of list A that is not already in,
        if its tails we put in the interleaved list the first element of list B that is not already in.

        @Input: 2 lists of integers of length 3.

        @Output: an interleaved list of tuples of length 3. Each tuple contain an integer (document ID) and a 0 or 1 
        depending of which list it came from.
    '''

    unique_doc_list = []
    interleaved_list = []
    counter = 0
    list_a_labeled, list_b_labeled = convert_lists_to_labeled(
        deepcopy(list_a), deepcopy(list_b))

    while counter < 3:
        coin_toss = rd.random()

        if(coin_toss > 0.5):
            put_first_available_url_in_interleaved(
                list_a_labeled, 0, interleaved_list, unique_doc_list)
            counter += 1

            if(counter == 3):
                return interleaved_list

            put_first_available_url_in_interleaved(
                list_b_labeled, 1, interleaved_list, unique_doc_list)
            counter += 1
        else:
            put_first_available_url_in_interleaved(
                list_b_labeled, 1, interleaved_list, unique_doc_list)
            counter += 1
            if(counter == 3):
                return interleaved_list

            put_first_available_url_in_interleaved(
                list_a_labeled, 0, interleaved_list, unique_doc_list)
            counter += 1

    print("List is:", interleaved_list)
    return interleaved_list


def put_first_available_url_in_interleaved(a_list, index_list, interleaved_list, unique_doc_list):
    ''' Helper method that creates a tuple with and integer (document ID) and 
        a 0 or 1 which indicates from which list it came from and adds it in the interleaved list. 

        @Input: - a list of intergers of length 3.
                - a 0 or 1 indicator of the list.
                - the interleaved list with all tuples of rnakings.

    '''

    for i in a_list:
        if i[1] not in unique_doc_list:
            interleaved_list.append((i[0], index_list))
            unique_doc_list.append(i[1])
            return


# %%
team_draft_interleaving(rankings[0][0], rankings[0][1])

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

        @Input: 2 lists of integers of length 3.

        @Output: an interleaved list of tuples of length 3. Each tuple contain an integer (document ID) and a 0 or 1 
        depending of which list it came from.

    '''
    interleaved_list = []
    counter = 0

    while counter < 3:
        coin_toss = rd.random()

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


def softmax(rankings, tau=3):
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
### Step 4: Implement Click-based models
'''

# %%


class YandexData():
    """
        The structure of lookup table:
        {
            'q_id':{
                'sessions':[
                {
                    'url_ids':[],
                    'clicks':[]
                }
                ]
                'docs':set([])
            }
        }

        @Properties:        
            - `q_id` is the id of the different queries, key to object:
                - `session` is a list of the sessions of this particular `q_id`
                    each `session` have a list of `url_ids` and a `clicks` list, 
                    the url in `clicks` must be in the `url_ids`
                - `docs` is the union of all urls returned by the system per query
    """

    def __init__(self, path):
        self.path = path
        self._load_data()

    def _load_data(self):
        CUTOFF = 3
        queries_lookup = {}

        # Lambda functions
        def new_item(): return {'sessions': [], 'docs': set()}

        def turn2int(x): return [int(i) for i in x]

        with open(self.path, 'r') as f:
            click = []
            last_q = None

            for line in f.readlines():
                vals = re.split(r'\t+', line.rstrip())

                # If line is a query
                if vals[2] == 'Q':
                    current_q = vals[3]

                    # Get the relevant URLs
                    cutoff_urls = turn2int(vals[5:5 + CUTOFF])

                    if current_q not in queries_lookup.keys():
                        it = new_item()
                    else:
                        it = queries_lookup[current_q]

                    # Append documents for this query and add session
                    it['docs'] = it['docs'].union(cutoff_urls)
                    it['sessions'].append({'urls': cutoff_urls, 'clicks': []})

                    queries_lookup[current_q] = it
                    last_q = current_q

                # Else if line is a click
                elif vals[2] == 'C':
                    # If the document has been found in the active query, add
                    # the document to the selection of clicks
                    if int(vals[3]) in queries_lookup[last_q]['sessions'][-1]['urls']:
                        queries_lookup[last_q][
                            'sessions'][-1]['clicks'].append(int(vals[3]))

        self.queries_lookup = queries_lookup

# %%


class ClickModel(object):
    def __init__(self):
        pass

    def train(self, data):
        raise NotImplementedError

    def get_probs(self, rankings):
        raise NotImplementedError

    def is_click(self, rankings, epsilon=0.1):
        """
            simulate the click, return a boolean list of the same length as `rankings`, True means clicked
        """
        probs = self.get_probs(rankings, epsilon)

        def click_fn(p): return rd.uniform(0, 1) < p
        return list(map(click_fn, probs))

# %%


class PBM(ClickModel):
    def __init__(self):
        self.alpha_uq = {}
        self.gamma_r = [rd.uniform(0, 1) for _ in range(CUTOFF)]

    def train(self, data, T=20, load=False):
        """ Trains the parameters of the model according to the data.

            @Input:
                - data: YandexData object, Yandex data
                - T: integer, time steps of the training loop
                - load: boolean, whether trained gamma is used (we dont need trained alpha because during inference they are replaced by epsilon)
        """

        if load:
            self.gamma_r = [0.9998564405092062,
                            0.48278049975990095, 0.3335993103977007]
            return

        self._init_alpha(data)

        for _ in range(T):
            self._update_alpha(data)
            self._update_gamma(data)

    def _init_alpha(self, data):
        """ Initializes alpha on the model to be a dictionary only where (query, doc) have a value

            @Input:
                - data: YandexData object
        """
        for q, it in data.queries_lookup.items():
            for doc in it['docs']:
                self.alpha_uq[(q, doc)] = rd.uniform(0, 1)

    def _update_alpha(self, data):
        """ Performs an update of alpha in EM

            @Input:
                - data: YandexData object
        """

        # Get all the sessions and docs belonging to a query.
        ql = data.queries_lookup
        new_alpha_uq = deepcopy(self.alpha_uq)

        # Loop through all query-doc combos
        for (q, u), alpha in self.alpha_uq.items():
            count = 2
            contribution_sum = 1

            for sess in ql[q]['sessions']:
                if u not in sess['urls']:
                    continue

                count += 1
                if u in sess['clicks']:
                    contribution_sum += 1
                else:
                    ind = sess['urls'].index(u)
                    contribution_sum += (1 - self.gamma_r[ind]) * \
                        alpha / (1 - self.gamma_r[ind] * alpha)

            new_alpha_uq[(q, u)] = contribution_sum / count

        self.alpha_uq = new_alpha_uq

    def _update_gamma(self, data):
        """ Performs an update of gamma in EM

            @Input:
                - data: YandexData object
        """

        ql = data.queries_lookup
        sess_num = 0
        contrib_sum = [0] * CUTOFF
        for q, item in ql.items():
            sess_num += len(item['sessions'])
            for sess in item['sessions']:
                for i, u in enumerate(sess['urls']):
                    if u in sess['clicks']:
                        contrib_sum[i] += 1
                    else:
                        contrib_sum[i] += self.gamma_r[i] * (1 - self.alpha_uq[q, u]) / (
                            1 - self.gamma_r[i] * self.alpha_uq[q, u])

        self.gamma_r = [i / sess_num for i in contrib_sum]
        return

    def get_probs(self, rankings, epsilon=0.1):
        """
            assume `rankings` are list of relevance labels
            use `epsilon` to substitute alpha, typical value is 0.1
        """
        def prob_fn(args): return self.gamma_r[
            args[0]] * (1 - epsilon if args[1] == 1 else epsilon)
        return list(map(prob_fn, enumerate(rankings)))


# %%
CUTOFF = 3


class RCM(ClickModel):
    """
        Random clicking model
    """

    def __init__(self):
        self.gamma = [0] * 3

    def train(self, data, load=True):
        """
            get the \rho paramter for random clicking by calculating the fraction of clicked urls among all returned results
        """
        if load:
            self.rho = 0.2802838475726031
            return
        sess_num = 0
        cli_num = 0
        for q, it in data.queries_lookup.items():
            for sess in it['sessions']:
                cli_num += len(sess['clicks'])
            sess_num += len(it['sessions'])
        self.rho = cli_num / sess_num / CUTOFF
        return

    def get_probs(self, rankings, epsilon=None):
        """
            return \rho list regardless
        """
        return [self.rho] * len(rankings)


# %%
'''
### Step 5: Simulate Interleaving Experiment
'''

# %%
# Import the appropriate data
yd = YandexData('./YandexRelPredChallenge.txt')

# Training the PBM and RCM model
model_PBM = PBM()
model_PBM.train(yd, 100, True)
model_RCM = RCM()
model_RCM.train(yd, True)

# %%
for i, bin_val in calculate_Dmeasures(rankings).items():
    for pair_idx in bin_val:
        couple = rankings[pair_idx]
        team_draft_interleaving(couple[0], couple[1])

# %%


def simulate_experiment(rankingA, rankingB, model, interleave_fn=team_draft_interleaving, k=100):
    E_wins = 0
    P_wins = 0

    for i in range(k):
        E_clicks = 0
        P_clicks = 0
        new_results_w_models = interleave_fn(rankingA, rankingB)
        new_results_relevance = [i[0] for i in new_results_w_models]
        ranker_clicked = [i[1] for i in new_results_w_models]
        clicks = model.is_click(new_results_relevance, 0.1)

        for index, click in enumerate(clicks):
            if click:
                if ranker_clicked[index] == 1:
                    E_clicks += 1
                else:
                    P_clicks += 1

        if E_clicks > P_clicks:
            E_wins += 1
        elif P_clicks > E_clicks:
            P_wins += 1

    return (E_wins + 1) / (E_wins + P_wins + 1)


# %%
'''
### Step 6: Estimate sample size
'''
def calc_sample_size(p_val, alpha=0.05, beta=0.10, p_null=0.5):
    z = norm.ppf(1-alpha)*math.sqrt(p_null * (1 - p_null)) + \
        norm.ppf(1-beta) * math.sqrt(p_val * (1-p_val))

    if p_val == p_null:
        return math.inf

    if z == 0.0:
        return -1

    return ((z/(abs(p_val-p_null)))**2) + 1/abs(p_val-p_null)


def calc_sample_size_for_bins(interleave_fn=team_draft_interleaving, model=model_PBM, rankings=rankings):
    bins = calculate_Dmeasures(rankings)
    bin_vals = list(bins.keys())
    table = pd.DataFrame(index=bin_vals,columns=['minimum', 'mean', 'maximum'])    

    for bin_key, bin_el in bins.items():
        minimum, mean, maximum = calc_sample_size_for_bin(bin_el, interleave_fn, model)
        table.loc[bin_key]['minimum'] = minimum
    
    return table


def calc_sample_size_for_bin(binned_el, interleave_fn, model):
    result = []

    for pair in binned_el:
        pairE = rankings[pair][0]
        pairP = rankings[pair][1]
        proportion_E_win = simulate_experiment(pairE, pairP, model, interleave_fn)
        sample_size = calc_sample_size(proportion_E_win)

        if sample_size >= 0:
            result.append(calc_sample_size(proportion_E_win))

    if len(binned_el) > 0:
        maximum = np.max(result)
        minimum = np.min(result)
        mean = np.mean(result)

        return (minimum, mean, maximum)
    
    return math.inf, math.inf, math.inf
        

table = calc_sample_size_for_bins()

# %%
'''
### Step 7: Analysis
'''
int_methods = [team_draft_interleaving, probabilistic_interleaving]

def run_all_setups(models=[RCM, PBM], methods=[team_draft_interleaving, probabilistic_interleaving]):
    for model in models:
        for method in methods:
            table_setup = calc_sample_size_for_bins(interleave_fn=method, model=model)
            # TODO: What should we do?
