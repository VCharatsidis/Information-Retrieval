import random as rd
from copy import deepcopy
import yandex

CUTOFF=3


class PBM(object):

    def __init__(self):
        self.alpha_uq = {}
        self.gamma_r = [rd.uniform(0, 1) for _ in range(CUTOFF)]

    def train(self, data, T=20, load=False):
        """ Trains the parameters according to data.

            @Input:
                - data: yandex data
                - T: time steps of the training loop
                - load: whether use the trained gamma(we dont need trained alpha because during inference they are replaced by epsilon).
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

    def is_click(self, rankings, epsilon):
        """
        simulate the click, return a boolean list of the same length as `rankings`, True means clicked
        """
        probs = self.get_probs(rankings, epsilon)
        click_fn = lambda p: rd.uniform(0, 1) < p
        return list(map(click_fn, probs))

if __name__ == "__main__":
    yd = yandex.YandexData('./YandexRelPredChallenge.txt')
    solver = PBM()
    solver.train(yd, 100, load=True)
    # print(list(solver.alpha_uq.items())[:2])
    print(solver.gamma_r)
    print(solver.get_probs([1,0,1], 1e-1))
    print(solver.is_click([1,0,1], 1e-1))
