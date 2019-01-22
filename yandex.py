import re


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
        new_item = lambda: {'sessions': [], 'docs': set()} 
        turn2int = lambda x: [int(i) for i in x] 
        
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
        return


if __name__ == "__main__":
    yd = YandexData('./YandexRelPredChallenge.txt')
    print(yd.queries_lookup['0'])