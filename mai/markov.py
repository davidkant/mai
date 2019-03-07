import random
import functools
import numpy as np
import matplotlib.pyplot as plt

class Markov:
    """A very simple markov model"""

    def __init__(self):

        # initialize transition table
        self.transitions = dict()

        # initialize current state
        self.state = ()

    def train(self, data, order=3, init_state=True, compute_matrix=False):
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

        # store transition table as matrix
        if compute_matrix:
            self.compute_transition_matrix()

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

    def choice(self, suppress_errors=False, k=None, include_initial=True):
        """Choose next value and allow for multiple choices at a time"""

        # store initial state
        initial_state = self.state

        # generate new sequence
        seq = self.choose(suppress_errors=suppress_errors) if k is None else \
            [self.choose(suppress_errors=suppress_errors) for i in range(k)]
        
        # prepend first order elements
        if include_initial:
            seq = list(initial_state) + seq

        return seq

    def clear(self):
        """Clear the transition table"""

        self.transitions.clear()

    def transition_matrix(self):
        """Transitions as a matrix"""

        return self.alpha

    def compute_transition_matrix(self):
        """Compute transitions as a matrix for internal use"""

        self.s1_states = sorted(set([x[0] for x in self.transitions.keys()]))
        self.s2_states = sorted(set([x[1] for x in self.transitions.keys()]))

        self.alpha = np.zeros((len(self.s1_states), len(self.s2_states)))

        for ((s1,s2), c) in self.transitions.items():
            s1_index = self.s1_states.index(s1)
            s2_index = self.s2_states.index(s2)
            self.alpha[s1_index, s2_index] = c

    def choose_np(self, suppress_errors=False):
        """Choose next value"""

        # if no successor found
        if self.state not in self.s1_states: 
            
            # option 1: raise error
            if not suppress_errors: raise LookupError('Current state not found in transition table')

            # option 2: random state from the entire table
            # not yet implemented

        # look up from state index
        s1_index = self.s1_states.index(self.state)

        # choose
        choice = random.choices(self.s2_states, weights=self.alpha[s1_index])[0]

        # update state
        self.state = tuple(list(self.state)[1:] + [choice])
                                
        # return
        return choice

    def choice_np(self, suppress_errors=False, k=None):
        """Choose next value and allow for multiple choices at a time"""

        return self.choose_np(suppress_errors=suppress_errors) if k is None else \
            [self.choose_np(suppress_errors=suppress_errors) for i in range(k)]

    def plot_transition_matrix(self, figsize=(3,3), figscale=0.5, cmap='viridis', show=True):
        """Plot transition matrix"""

        self.compute_transition_matrix()
        alpha = self.transition_matrix()
        num_rows, num_cols = alpha.shape
        figscale = figscale
        fig = plt.figure(figsize=(figscale*num_cols, figscale*num_rows))
        ax = plt.gca()
        ax.imshow(alpha, cmap=cmap, aspect='auto')
        ax.grid(False)
        if show: 
            plt.show()
        else:
            return fig

    def print_transitions(self):
        """Pretty print transition tabale"""
        for k,v in self.transitions.items():
            print("{0} -> {1} : {2}".format(k[0], k[1], v))
