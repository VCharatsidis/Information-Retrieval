import random as rd
from copy import deepcopy
import yandex

CUTOFF=3

class PBM(object):
    def __init__(self):
        self.alpha_uq = {}
        self.gamma_r = [rd.uniform(0.25, 0.75) for _ in range(CUTOFF)]

    def train(self, data, T=20):
        self._init_alpha(data)
        for _ in range(T):
            self._update_alpha(data)
            self._update_gamma(data)


    def _init_alpha(self, data):
        for q, it in data.queries_lookup.items():
            for doc in it['docs']:
                self.alpha_uq[(q, doc)] = rd.uniform(0.25, 0.75)

    def _update_alpha(self, data):
        ql = data.queries_lookup
        new_alpha_uq = deepcopy(self.alpha_uq)
        for (q, u), alpha in self.alpha_uq.items():
            count = 0
            contribution_sum = 0
            for sess in ql[q]['sessions']:
                if u not in sess['urls']:
                    continue
                count+=1
                if u in sess['clicks']:
                    contribution_sum+=1
                else:
                    ind = sess['urls'].index(u)
                    contribution_sum+=(1-self.gamma_r[ind])*alpha/(1-self.gamma_r[ind]*alpha)
            new_alpha_uq[(q, u)] = contribution_sum/count
        self.alpha_uq = new_alpha_uq
        return

    def _update_gamma(self, data):
        ql = data.queries_lookup
        sess_num = 0
        contrib_sum = [0]*CUTOFF
        for q, item in ql.items():
            sess_num += len(item['sessions'])
            for sess in item['sessions']:
                for i, u in enumerate(sess['urls']):
                    if u in sess['clicks']:
                        contrib_sum[i]+=1
                    else:
                        contrib_sum[i]+=self.gamma_r[i]*(1-self.alpha_uq[q, u])/(1-self.gamma_r[i]*self.alpha_uq[q, u])

        self.gamma_r = [i / sess_num for i in contrib_sum]
        return 


    def get_probs(rankings):
        pass
    def is_click(doc):
        pass

if __name__ == "__main__":
    yd = yandex.YandexData('./YandexRelPredChallenge.txt')
    solver = PBM()
    solver.train(yd)
    print(list(solver.alpha_uq.items())[:2])
    print(solver.gamma_r)
