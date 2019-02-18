import numpy as np
import random

# Core CA ---------------------------------------------------------------------

def apply_rule(triplet, rule):
    """Apply an update rule to a single cell (taken together with its neighbors)."""
    
    # interpret triplet as binary number
    index = 7 - int("".join(map(str, triplet)), 2)
    
    # lookup rule index and return
    return rule[index]


def update_gen(gen, rule, wrap=True, edge=True):
    """Apply an update rule to an entire generation.
    
    Parameters
    ----------
    wrap : bool
        Whether or not CA neighbors wrap around edges.
        
    edge : bool
        Whether or not to include edge cells.
    """  

    # apply rule and return
    if not edge:
        return [gen[0]] + [apply_rule(xyz, rule) for xyz in zip(gen[:-2], gen[1:-1], gen[2:])] + [gen[-1]]

    # either zero pad or wrap depending on wrap
    gen_minus_1 = [0] + gen[:-1] if not wrap else gen[-1:] + gen[:-1]
    gen_plus_1 = gen[1:] + [0] if not wrap else gen[1:] + gen[-1:]

    # apply rule and return
    return [apply_rule(xyz, rule) for xyz in zip(gen_minus_1, gen, gen_plus_1)]


def generate(rule, size=31, iters=15, wrap=True, edge=True, init_pop=None, init_random=False):
    """Generate a 1d cellular automata.
    
    Parameters
    ----------
    rule : list
        Rule set represented as list of 8 values, each either a 0 or 1.

    size : int
        Number of cells in a generation. Should be odd if initialized to center on all else off.
    
    iters : int
        Number of generations to generate.
        
    wrap : bool
        Whether or not CA neighbors wrap around edges.

    edge : bool
        Whether or not to include edge cells.

    initial_pop : list
        Use this to supply an initial population. Overrides supplied size argument. Should be list of 0's and 1's.
    
    init_random : bool
        Initialize initial population to random if true otherwise to middle cell on all else off.
    
    Returns
    -------
    gens : list of lists
        List of lists in which each list represents a generation.
    
    """
    
    # initial generation
    gen = init_pop if init_pop is not None \
        else [random.randint(0, 1) for i in range(size)] if init_random \
        else [0]*int(size/2) + [1] + [0]*int(size/2)

    # iterate multiple generations
    gens = [gen]
    for i in range(iters):
        gens.append(update_gen(gens[-1], rule, wrap=wrap))
        
    # return as list of lists
    return gens


# Plotters --------------------------------------------------------------------

import matplotlib.pyplot as plt

def plot(gens):
    """Plot a set of 1d CA generations."""
    fig, ax = plt.subplots(figsize=(9,9))
    ax.imshow(np.array(gens))
    ax.set_xticks(np.arange(0, len(gens[0]), 1));
    ax.set_yticks(np.arange(0, len(gens), 1));
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='grey', linestyle='-', linewidth=1)
    plt.show()


# Animate ---------------------------------------------------------------------

from matplotlib import animation, rc

def animate(gens, interval=200):
    """Animate a set of 1d CA generations."""
    
    # we'll need these
    num_cols = len(gens[0])
    num_rows = len(gens)
    
    # set up figure
    fig, ax = plt.subplots(figsize=(num_cols*0.3, num_rows*0.3))
    plt.close()
    ax.set_xlim([-1, num_cols])
    ax.set_xticks(np.arange(0,num_cols+1,1)-0.5)
    ax.set_xticklabels([])
    ax.set_ylim([0, num_rows+1])
    ax.set_yticks(np.arange(1,num_rows+2,1)-0.5)
    ax.set_yticklabels([])
    line, = ax.plot([],[], 's', c='k', markersize=15)
    
    # update function
    data = []
    def update(i, data):
        # current row
        row = gens[i]    
        # filter for cells that are on
        new_data = [index for index,val in enumerate(row) if val == 1]
        # offset y coord by current row
        new_data = [(index,num_rows-i) for index in new_data]
        # append to data
        data += new_data
        # update plot data
        line.set_data([x[0] for x in data], [x[1] for x in data])
        # return
        return line,

    # animation
    anim = animation.FuncAnimation(fig, update, fargs=(data,), frames=num_rows, interval=interval, blit=True)

    # run it as html
    rc('animation', html='jshtml')
    return anim


# Synthesis -------------------------------------------------------------------

import IPython.display
import librosa

def synth_waveform(gens, stride=1, format='inbrowser', sr=44100):
    """Synthesize CA gens as a waveform.
    
    Parameters
    ----------
    gens : list of lists
        CA generations as a list of lists
    stride : int
        Samples per CA cell
    format : string
        Which format to render sound to?
            - `'audio'` returns waveforms as a `numpy` nd.array  
            - `'inbrowser'` returns `IPython.display.Audio` widget 
            - `'autoplay'` returns `IPython.display.Audio` widget and plays it
    sr : int
        Sample rate
    
    Returns
    -------
    y : depends on the value of `format`
    """
    
    # flatten to one list and convert to numpy
    y = np.ravel(gens)

    # stretch by stride
    y = np.repeat(y, stride)

    # return
    if format=='audio':
        return y
    elif format=='inbrowser':
        return IPython.display.Audio(y, rate=sr)
    elif format=='autoplay':
        return IPython.display.Audio(y, rate=sr, autoplay=True)
    else:
        raise ValueError("So sorry but your `format` argument did not match one of the available options")


def plot_waveform(gens, stride=1, figsize=(9,3)):
    """Plot CA gens as a waveform."""

    # flatten to one list and convert to numpy
    y = np.ravel(gens)
    
    # stretch by stride
    y = np.repeat(y, stride)

    # plot
    plt.figure(figsize=figsize)
    plt.plot(y)
    plt.show()
    

def synth_iFFT(gens, stride=1, format='inbrowser', sr=44100):
    """Synthesize CA gens using iFFT synthesis.
    
    Parameters
    ----------
    gens : list of lists
        CA generations as a list of lists
    stride : int
        Frames per CA generation
    format : string
        Which format to render sound to?
            - `'audio'` returns waveforms as a `numpy` nd.array  
            - `'inbrowser'` returns `IPython.display.Audio` widget 
            - `'autoplay'` returns `IPython.display.Audio` widget and plays it
    sr : int
        Sample rate
    
    Returns
    -------
    y : depends on the value of `format`
    """
    
    # convert to numpy array 2d
    Y = np.array(gens) * 33
    
    # stretch by stride
    Y = np.repeat(Y, stride, axis=0)
    
    # inverse FFT synthesis
    y = librosa.istft(Y)

    # return
    if format=='audio':
        return y
    elif format=='inbrowser':
        return IPython.display.Audio(y, rate=sr)
    elif format=='autoplay':
        return IPython.display.Audio(y, rate=sr, autoplay=True)
    else:
        raise ValueError("So sorry but your `format` argument did not match one of the available options")
        

def plot_iFFT(gens, stride=1, waveform=False):
    """Plot CA gens as using iFFT synthesis."""

    # convert to numpy array 2d
    Y = np.array(gens) * 33
    
    # stretch by stride
    Y = np.repeat(Y, stride, axis=0)

    # inverse FFT synthesis
    y = librosa.istft(Y)
    
    # plot waveform
    if waveform:
        plt.figure(figsize=(12,2))
        plt.plot(y)
        plt.show()

    # plot FFT spectrum
    plt.figure(figsize=(12,4))
    plt.imshow(Y.T, origin='lowerleft', aspect='auto')
    plt.show()
