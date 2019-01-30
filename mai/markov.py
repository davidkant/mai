import random
import functools

class Markov:
    """A very simple markov model"""

    def __init__(self):

        # initialize transition table
        self.transitions = dict()

        # initialize current state
        self.state = ()

    def train(self, data, order=3, init_state=True):
        """Train the Markov model"""

        # loop through data
        for i in range(order, len(data)):

            # previous state   
            prev_state = tuple(data[i-order:i])

            # current state 
            curr_state = data[i]

            # key is a tuple: (prev, curr)
            key = (prev_state, curr_state)

            # if key in transitions
            if key in self.transitions:

                # increment count by 1
                self.transitions[key] += 1

            # if key not in transitions
            else:
                                                                                        
                # initialize count to 1
                self.transitions[key] = 1

        # set initial state
        if init_state:
            self.state = tuple(data[:order])

    def choose(self, suppress_errors=False):
        """Choose next value"""

        # get all m-grams for the current state
        alpha = dict(filter(lambda kv: kv[0][0]==self.state, self.transitions.items()))

        # if no successor found
        if len(alpha) == 0: 
            
            # option 1: raise error
            if not suppress_errors: raise LookupError('Current state not found in transition table')
        
            # option 2: random state from the entire table
            else: alpha = self.transitions 

        # weighed choose
        choice = random.choice(functools.reduce(lambda x,y: x+y, [[k]*v for k,v in alpha.items()]))

        # update state
        self.state = tuple(list(self.state)[1:] + [choice[1]])
                                
        # return
        return choice[1]

    def choice(self, suppress_errors=False, k=None):
        """Choose next value and allow for multiple choices at a time"""

        return self.choose(suppress_errors=suppress_errors) if k is None else \
            [self.choose(suppress_errors=suppress_errors) for i in range(k)]

    def clear(self):
        """Clear the transition table"""

        self.transitions.clear()
