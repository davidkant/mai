import numpy as np
from . import musifuncs as mf

class SinOsc():
    """A simple sine wave oscillator."""

    def __init__(self, freq=440, iphase=0, sr=44100):
        self.TWO_PI = np.pi * 2
        self.freq = freq
        self.phase = iphase * self.TWO_PI % self.TWO_PI
        self.sr = sr

    def next(self):
        out = np.sin(self.phase)
        self.phase += self.freq * self.TWO_PI / self.sr % self.TWO_PI
        np.sin(1*2*np.pi)
        return out

class Ramp():
    """A simple ramp envelope generator."""
  
    def __init__(self, x1=1, x2=0, dur=1, sr=44100):
        self.x1 = x1
        self.x2 = x2
        self.dur = dur
        self.sr = sr
        self.buff = np.linspace(self.x1, self.x2, num=int(float(self.dur) * float(self.sr)))
        self.index = 0
    
    def next(self):
        out = self.buff[self.index]
        self.index += 1 
        return out
  
class ADSR():
    """A very simple ADSR evenelope generator."""
    
    def __init__(self, attack=1, decay=0, sustain=0, release=1, sr=44100):
        self.attack = attack
        self.decay = decay
        self.susatin = sustain
        self.release = release
        self.sr = sr
        self.env_attack = np.linspace(0, 1, num=int(float(self.attack) * float(self.sr)))
        self.env_release = np.linspace(1, 0, num=int(float(self.release) * float(self.sr)))
        self.buff = np.concatenate((self.env_attack, self.env_release))
        self.index = 0
    
    def next(self):
        out = self.buff[self.index]
        self.index += 1 
        return out
 
class FM1():
    """A FM synth with a few control parameters."""

    def __init__(self, carrier=900, modulator=300, index1=5, index2=0, attack=1, release=1, sr=44100):
        self.carrier = carrier
        self.modulator =  modulator
        self.index1 = index1
        self.index2 = index2
        self.attack = attack
        self.release = release
        self.sr = sr
        self.osc1 = SinOsc(self.modulator)
        self.osc2 = SinOsc(self.carrier)
        self.index = Ramp(index1, index2, self.attack+self.release)
        self.adsr = ADSR(self.attack, 0, 0, self.release)
    
    def next(self):
        self.osc2.freq = self.osc1.next() * float(self.index.next()) * float(self.modulator) + float(self.carrier)
        return self.osc2.next() * self.adsr.next()

    def render(self):
        dur = int(self.sr * self.attack) + int(self.sr * self.release)
        return np.array([self.next() for i in range(dur-1)])

def test_FM1():
    carrier = 900
    modulator = 300
    index1 = 10
    index2 = 0
    attack = 1
    release = 0
    sr = 44100
    fm = FM1(carrier, modulator, index1, index2, attack, release)
    return np.array([fm.next() for i in range(int(44100*(attack+release)))])

def test_FM1_render():
    carrier = 900
    modulator = 300
    index1 = 10
    index2 = 0
    attack = 1
    release = 0
    sr = 44100
    fm = FM1(carrier, modulator, index1, index2, attack, release)
    return fm.render()
