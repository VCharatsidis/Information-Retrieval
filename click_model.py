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

