import random

class FSTN:
    """A very simply finite state transition network."""

    def __init__(self):

        # initialize edges to empty list
        self.edges = []

    def add_edge(self, s1, s2, label):
        """Add an edge from s1 to s2 with label."""

        # add edge to list of edges
        self.edges += [[s1, s2, label]]

    def walk(self):
        """Walk the FSTN.
        
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
            # add transition label to the walk
            if next_label != []:
                walk += [next_label]
            # update current state
            curr_state = next_state

        # return it
        return walk[:-1]

def pretty_print(walk):
    """Pretty print a walk."""

    return reduce(lambda x,y: x + ' ' + y, walk)

def test_FSTN():
    """Test using a simple FSTN."""

    # create FSTN
    g = FSTN()

    # add edges
    g.add_edge(1, 2, 'a')
    g.add_edge(2, 3, 'rose')
    g.add_edge(3, 1, 'is')
    g.add_edge(3, 0, '')

    # run FSTN
    walk = g.walk()
    print(walk)
    print(pretty_print(walk))
