import random


def amb(pitch_center=40, pitch_range=6, pulse=120, rhythm=0.0, detune=0.0, repeat=0.5, memory=5, length=24):
    """Simple version of Larry Polansky's Anna's Music Box as a function."""

    # start with empty lists for both pitch and duration
    my_pitches = []
    my_durs = []

    # loop until we have enough notes
    while len(my_durs) < length:

        # do we look back?
        if random.random() < repeat and len(my_pitches) > memory:

            # use the fifth previous note
            new_pitch = my_pitches[-memory]
            new_dur = my_durs[-memory]

        # if we don't look back
        else:

            # choose pitch
            new_pitch = random.randint(pitch_center - pitch_range, pitch_center + pitch_range)

        # microtonal pitch adjustment 
        new_pitch += random.uniform(-detune, detune)

        # choose duration
        new_dur = (60.0 / pulse) * random.uniform(1-rhythm, 1+rhythm)
    
        # append to the melody
        my_pitches += [new_pitch]
        my_durs += [new_dur]
    
    return my_pitches, my_durs
