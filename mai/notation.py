import abjad
import abjadext.nauert


def make_music_notation(pitches=60, durs=0.333, tempo=60, time_signature=(4,4)):
    """Quick and simple music notation with Abjad.

    Converts pitch and duration values into music notation. Uses the Abjad
    quantizer.
        
    Parameters
    ----------
    pitches : list or scalar
        List of pitches, or scalar if constant pitch. Floating point values are
        interpreted as microtonal pitch deviations.
    durs: list or scalar
        List of durations, or scalar if constant duration. 
    tempo: number
        Quarter note tempo mark.
    time_signature : tuple
        Musical time signature: (beats per measure, which note gets the beat).

    Returns
    -------
    abjadext.ipython display 

    Notes
    -----
    If len(pitches) and len(durs) do not match, the smaller list is extended to 
    match the length of the longer list by repeating the last value.

    Requires ipython ``%load_ext abjadext.ipython`` for display
    """

    # check and convert to list if needed
    pitches = pitches if isinstance(pitches, list) else [pitches]
    durs = durs if isinstance(durs, list) else [durs]
  
    # extend short lists if size mismatch
    max_length = max(len(pitches), len(durs))
    pitches += [pitches[-1]] * (max_length - len(pitches))
    durs += [durs[-1]] * (max_length - len(durs))

    # offset pitches to C4 = 60
    pitches = [p - 60 for p in pitches]

    # scale durations to milliseconds
    durs = [d * 1000.0 for d in durs]
  
    # construct schema from tempo and time signature
    q_schema = abjadext.nauert.MeasurewiseQSchema(
        tempo=abjad.MetronomeMark((1, 4), tempo),
        time_signature=abjad.TimeSignature(time_signature),
        use_full_measure=True,
    )

    # sequence to be quantized
    q_event_seq = abjadext.nauert.QEventSequence.from_millisecond_pitch_pairs(tuple(zip(durs, pitches)))

    # quantize
    quantizer = abjadext.nauert.Quantizer()
    result = quantizer(q_event_seq, q_schema=q_schema)

    # stuff staff and score
    staff = abjad.Staff([result])
    score = abjad.Score([staff])

    # show
    return abjad.show(score)
