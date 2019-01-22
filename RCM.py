import random as rd
from click_model import ClickModel
import yandex

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
if __name__ == "__main__":
    yd = yandex.YandexData('./YandexRelPredChallenge.txt')
    solver = RCM()
    solver.train(yd)
    solver.train
    print(solver.get_probs([1, 0, 1], 1e-1))
    print(solver.is_click([1, 0, 1], 1e-1))
