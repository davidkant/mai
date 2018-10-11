import pretty_midi
import IPython.display
import math
import matplotlib.pyplot as plt


def make_music(pitches=60, durs=0.333, pgm=1, is_drum=False, format='inbrowser', sr=16000):
    """Turn lists of numbers into music.

    Converts pitch and duration values into MIDI and/or audio playback. Uses
    `pretty_midi` for MIDI representation handling, fluidsynth for resynthesis, 
    and `IPython.display.Audio` for browser playback.
        
    Parameters
    ----------
    pitches : list or scalar
        List of pitches, or scalar if constant pitch. Floating point values are
        interpreted as microtonal pitch deviations.
    durs: list or scalar
        List of durations, or scalar if constant duration. 
    pgm: number
        MIDI program number, in range ``[0, 127]``.
    is_drum : bool
        If True use percussion channel 10.
    format : string
        Which format to render sound to?
        - `'MIDI'` returns MIDI as a `pretty_midi` object
        - `'audio'` returns waveforms as a `numpy` nd.array  
        - `'inbrowser'` returns `IPython.display.Audio` widget 
        - `'autoplay'` returns `IPython.display.Audio` widget and plays it

    Returns
    -------
    synthesized: depends on the value of `format`.

    Notes
    -----
    If len(pitches) and len(durs) do not match, the smaller list is extended to 
    match the length of the longer list by repeating the last value.
    """

    # check and convert to list if needed
    pitches = pitches if isinstance(pitches, list) else [pitches]
    durs = durs if isinstance(durs, list) else [durs]
  
    # extend short lists if size mismatch
    max_length = max(len(pitches), len(durs))
    pitches += [pitches[-1]] * (max_length - len(pitches))
    durs += [durs[-1]] * (max_length - len(durs))
  
    # create a PrettyMIDI score
    score = pretty_midi.PrettyMIDI()

    # create an instrument
    ins = pretty_midi.Instrument(program=pgm-1, is_drum=is_drum)

    # iterate through music
    now_time = 0
    for pitch,dur in zip(pitches, durs):

        # split into 12tet and microtones
        micros, pitch = math.modf(pitch)

        # create a new note
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=now_time, end=now_time+dur)

        # and add it to the instrument
        ins.notes.append(note)

        # if microtonal
        if micros != 0:
                                  
            # create a new pitch bend
            # note: 4096 is a semitone in standard MIDI +/-2 pitchbend range
            micropitch = pretty_midi.PitchBend(pitch=int(round(micros*4096)), time=now_time)

            # and add it to the instrument
            ins.pitch_bends.append(micropitch)

        # advance time
        now_time += dur

    # add instrument to the score
    score.instruments.append(ins)

    # which format to render
    if format=='MIDI':
        return score 
    elif format=='audio':
        return score.fluidsynth(fs=16000) 
    elif format=='inbrowser':
        return IPython.display.Audio(score.fluidsynth(fs=sr), rate=sr)
    elif format=='autoplay':
        return IPython.display.Audio(score.fluidsynth(fs=sr), rate=sr, autoplay=True)
    else:
        raise ValueError("So sorry but your `format` argument did not match one of the available options")

def make_music_plot(pitches=60, durs=0.333, pgm=1, is_drum=False, format='autoplay', sr=16000, figsize=(9,3), cmap='jet', show=True):
    """Plot lists of numbers as music (same API as `make_music`)"""

    # check and convert to list if needed
    pitches = pitches if isinstance(pitches, list) else [pitches]
    durs = durs if isinstance(durs, list) else [durs]
  
    # extend short lists if size mismatch
    max_length = max(len(pitches), len(durs))
    pitches += [pitches[-1]] * (max_length - len(pitches))
    durs += [durs[-1]] * (max_length - len(durs))

    # plot
    plt.figure(figsize=figsize)
    cm = plt.cm.get_cmap(name=cmap)
    curr_time = 0
    for pitch,dur in zip(pitches, durs):
        pitch_normed = float(pitch - min(pitches)) / (max(pitches) - min(pitches)) if (max(pitches) - min(pitches)) != 0 else 1
        plt.scatter([curr_time], [pitch], marker='|', c='white', s=25, zorder=3)
        plt.plot([curr_time, curr_time + dur], [pitch, pitch], lw=5, solid_capstyle='butt', c=cm(pitch_normed), alpha=0.75)
        curr_time += dur

    if show:
        plt.show()
