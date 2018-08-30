import librosa
import essentia
import numpy as np

def spectral_features(filename, frameSize=2048, hopSize=1024, sr=44100):
    """Extract basic spectral features."""

    # if y is audio write a temp file for essentia to load
    if isinstance(filename, np.ndarray):
      librosa.output.write_wav('tempaudio.wav', y, sr=sr, norm=False)
      filename = 'tempaudio.wav'

    # features to return
    basic_features_names = [ 
        'lowlevel.spectral_centroid',
        'lowlevel.spectral_complexity',
        'lowlevel.spectral_crest',
        'lowlevel.spectral_decrease',
        'lowlevel.spectral_energy',
        'lowlevel.spectral_energyband_high',
        'lowlevel.spectral_energyband_low',
        'lowlevel.spectral_energyband_middle_high',
        'lowlevel.spectral_energyband_middle_low',
        'lowlevel.spectral_entropy',
        'lowlevel.spectral_flatness_db',
        'lowlevel.spectral_flux',
        'lowlevel.spectral_kurtosis',
        'lowlevel.spectral_rms',
        'lowlevel.spectral_rolloff',
        'lowlevel.spectral_skewness',
        'lowlevel.spectral_spread',
        'lowlevel.spectral_strongpeak'
  ]

    # create feature extractor
    freesound_extractor = essentia.standard.FreesoundExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev'],
        analysisSampleRate=sr,
        lowlevelFrameSize=frameSize,
        lowlevelHopSize=hopSize
    )

    # extract features
    features, features_frames = freesound_extractor(filename)

    # filter a dict of basic features
    basic_features = dict([(feature_name, features_frames[feature_name]) for feature_name in basic_features_names])

    return basic_features
