# %%

import numpy as np


# %%
'''

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


def calculate_ERR(ranking):
    ''' This method calculates the ERR of a ranking. A ranking is a one dimensional interger list with length 3.
    
        @Input: a one dimensional list of length 3 with zeros and ones.
        
        @Output: a double (ERR score).
    '''
    ERR = 0
    for r in range(len(ranking)):
        prob_to_stop_at_r = ranking[r] / (r + 1)
        for i in range(r):
            prob_to_stop_at_r *= 1 - ranking[i]

        ERR += prob_to_stop_at_r

    return ERR

calculate_ERR(rankings[1][1])

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
    ''' Team draft interleaving performed by throwing a coin. If its heads we put in the interleaved list
        the first element of list A that is not already in,
        if its tails we put in the interleaved list the first element of list B that is not already in.
        
        @Input: 2 lists of intergers of length 3.
        
        @Output: an interleaved list of tuples of length 3. Each tuple contain an integer (document ID) and a 0 or 1 
        depending of which list it came from.
    
    '''

    interleaved_list = []
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

        if i not already_in_interleaved:
            tup = (i, index_list)
            intearleaved_list.append(tup)
            return


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
    interleaved_list = []
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

# %%
import re

CUTOFF = 3
class YandexData():
    """
    take the path to YandexData, load all needed entries.
    """
    def __init__(self, path):
        self.path = path
        self._load_data()

    def _load_data(self):
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
        `q_id` is the id of the different queries
        `session` is a list of the sessions of this particular `q_id`
        each `session` have a list of `url_ids` and a `clicks` list, the url in `clicks` must be in the `url_ids`
        `docs` is the union of all urls returned by the system per query
        """
        queries_lookup = {}
        new_item = lambda: {'sessions': [], 'docs': set()}
        turn2int = lambda x: [int(i) for i in x]
        # state='Q'
        with open(self.path, 'r') as f:
            click = []
            last_q = None
            for line in f.readlines():
                vals = re.split(r'\t+', line.rstrip())
                if vals[2] == 'Q':
                    # state = 'Q'
                    pres_q = vals[3]
                    # we only care about the first CUTOFF urls
                    cutoff_urls = turn2int(vals[5:5 + CUTOFF])
                    if pres_q not in queries_lookup.keys():
                        it = new_item()
                    else:
                        it = queries_lookup[pres_q]
                    it['docs'] = it['docs'].union(cutoff_urls)
                    it['sessions'].append({'urls': cutoff_urls, 'clicks': []})
                    queries_lookup[pres_q] = it
                    last_q = pres_q
                elif vals[2] == 'C':
                    if int(vals[3]) in queries_lookup[last_q]['sessions'][-1]['urls']:
                        queries_lookup[last_q][
                            'sessions'][-1]['clicks'].append(int(vals[3]))
                    # state = 'C'
        self.queries_lookup = queries_lookup
        return

# %%
import random as rd

class ClickModel(object):

    def __init__(self):
        pass
    def train(self, data):
        raise NotImplementedError
    def get_probs(self, rankings):
        raise NotImplementedError
    def is_click(self, rankings, epsilon):
        """
        simulate the click, return a boolean list of the same length as `rankings`, True means clicked
        """
        probs = self.get_probs(rankings, epsilon)
        click_fn = lambda p: rd.uniform(0, 1) < p
        return list(map(click_fn, probs))

# %%


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
from copy import deepcopy

class PBM(ClickModel):
    """
        Position based clicking model
    """
    def __init__(self):
        self.alpha_uq = {}
        self.gamma_r = [rd.uniform(0, 1) for _ in range(CUTOFF)]

    def train(self, data, T=20, load=False):
        """
            train the parameter according to data
            data: yandex data
            T: time steps of the training loop
            load: whether use the trained gamma(we dont need trained alpha because during inference they are replaced by epsilon)
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
        """
        init the alpha to be a dict where only the present (query, doc) pairs have value
        """
        for q, it in data.queries_lookup.items():
            for doc in it['docs']:
                self.alpha_uq[(q, doc)] = rd.uniform(0, 1)

    def _update_alpha(self, data):
        """
        data is the yandex data.
        This function performs an update of alpha in EM
        """

        ql = data.queries_lookup
        new_alpha_uq = deepcopy(self.alpha_uq)
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
        return

    def _update_gamma(self, data):
        """
        data is the yandex data.
        This function performs an update of gamma in EM
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

    def get_probs(self, rankings, epsilon):
        """
            assume `rankings` are list of relevance labels
            use `epsilon` to substitute alpha, typical value is 0.1
        """
        prob_fn = lambda args: self.gamma_r[
            args[0]] * (1 - epsilon if args[1] == 1 else epsilon)
        return list(map(prob_fn, enumerate(rankings)))
