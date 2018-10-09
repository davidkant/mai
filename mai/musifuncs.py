import math
import numpy as np

# converters ------------------------------------------------------------------

def ratio_to_cents(r): 
    """Pitch ratio to cents."""
    return 1200.0 * np.log2(r)

def cents_to_ratio(c): 
    """Cents to pitch ratio."""
    return np.power(2, c/1200.0)

def hz_to_midi(f): 
    """Frequency in Hz to midi note number."""
    return 69.0 + 12.0 * np.log2(f/440.0)

def midi_to_hz(m): 
    """Midi note number to frequency."""
    return np.power(2, (m-69.0)/12.0) * 440.0 if m!= 0.0 else 0.0

def midi_to_cents(m):
    """Split 12tet and cents."""  
    p = np.round(m)
    c = (m - p)* 100
    return p,c
   
# interpolation ---------------------------------------------------------------

def linear_map(val, lo, hi):
    """Linear mapping."""
    return val * (hi - lo) + lo

def linear_unmap(val, lo, hi):
    """Linear unmapping."""
    return (val - lo) / (hi - lo)

def exp_map(val, lo, hi):
    """Exponential mapping."""
    return pow(hi / lo, val) * lo

def exp_unmap(val, lo, hi):
    """Exponential unmapping."""
    return math.log(val / lo) / math.log(hi / lo)

# spec ------------------------------------------------------------------------

class ControlSpec: 
    """A very basic SC-style control spec."""

    def __init__(self):
        self.data = dict()

    def add(self, param, spec):
        """Store in dict."""
        self.data[param] = spec

    def map_spec(self, param, val):
        """Map from normal."""
        lo, hi, curve = self.data[param]
        val = self.clip(val, 0.0, 1.0)
        if curve is 'linear':
            return self.linear_map(float(val), float(lo), float(hi))
        if curve is 'exp':
            return self.exp_map(float(val), float(lo), float(hi))

    def unmap_spec(self, param, val):
        """Unmap to normal."""
        lo, hi, curve = self.data[param]
        clip_lo = min(lo, hi)
        clip_hi = max(lo, hi)
        val = self.clip(val, clip_lo, clip_hi)
        if curve is 'linear':
            return self.linear_unmap(float(val), float(lo), float(hi))
        if curve is 'exp':
            return self.exp_unmap(float(val), float(lo), float(hi))

    def linear_map(self, val, lo, hi):
        """Linear mapping."""
        return val * (hi - lo) + lo

    def linear_unmap(self, val, lo, hi):
        """Linear unmapping."""
        return (val - lo) / (hi - lo)

    def exp_map(self, val, lo, hi):
        """Exponential mapping."""
        return pow(hi / lo, val) * lo

    def exp_unmap(self, val, lo, hi):
        """Exponential unmapping."""
        return math.log(val / lo) / math.log(hi / lo)

    def clip(self, val, lo, hi):
        """Clip to hi and lo."""
        return lo if val < lo else hi if val > hi else val
