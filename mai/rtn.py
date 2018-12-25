import random

class RTN:
    """A very simply recursive transition network.

    * non-termal edges are represented as callable
    * unlabelled edges are represented as []
    * terminal edges are represnted as symbols
    """

    def __init__(self):

        # initialize edges to empty list
        self.edges = []

    def add_edge(self, s1, s2, label):
        """Add an edge from s1 to s2 with label."""

        # add edge to list of edges
        self.edges += [[s1, s2, label]]

    def walk(self):
        """Walk the RTN.

        note: 
          * start on state one by default
          * walk until state 0
          * it is very possible to enter an inifnite loop if
            - there is no final state 0 to end the network
            - if we land on a state that has no transitions
        """

        # initialize current state and walk
        curr_state = 1
        walk = []

        # walk until land on state 0
        while curr_state != 0:
            # possible choices for next state
            choices = [(y, label) for x,y,label in self.edges if x==curr_state]
            # choose a next state
            next_state, next_label = random.choice(choices) if choices != [] else curr_state
            # add label if not empty transition
            if next_label != []:
                # evaluate label if callable (non-terminal) 
                label = next_label() if callable(next_label) else next_label
                # append it to the list
                walk += label if isinstance(label, list) else [label]
            # update current state
            curr_state = next_state

        # return it
        return walk[:-1]

    def __call__(self):
        return self.walk()

def pretty_print(walk):
    """Pretty print a walk."""

    return reduce(lambda x,y: x + ' ' + y, walk)

def test_RTN():
    """Test using a simple RTN."""

    # create RTN
    g = RTN()

    # define non-terminal symbols
    def V():
        return "eats"

    def WH():
        return random.choice(['who', 'whose'])

    def PN():
        return random.choice(["George", "cheese"])

    # NP subnetwork
    NP = RTN()
    NP.add_edge(1, 2, WH)
    NP.add_edge(1, 2, PN)
    NP.add_edge(2, 0, '')

    # add edges
    g.add_edge(1, 2, NP) # non-termal edges represented as callable
    g.add_edge(2, 1, []) # unlabelled edges represented as []
    g.add_edge(2, 3, V)
    g.add_edge(3, 1, [])
    g.add_edge(3, 4, NP)
    g.add_edge(4, 2, [])
    g.add_edge(4, 0, '') # terminal edges represnted as symbols

    # run FSTN
    walk = g.walk()
    print(walk)
    print(pretty_print(walk))
