import librosa
import numpy as np

# feature extraction ----------------------------------------------------------

def extract_mfcc(y, sr, **kwargs):
    return librosa.feature.mfcc(y, sr=sr, n_mfcc=20, **kwargs)

def extract_magspec(y, sr, **kwargs):
    return librosa.magphase(librosa.stft(y, **kwargs))[0]

def extract_cqt(y, sr, **kwargs):
    return librosa.cqt(y, sr=sr)

# metrics ---------------------------------------------------------------------

def cosine_distance(A,B, norm=True):
    """Cosine distance between two spectrograms [bins, frames]."""

    A_norm = np.linalg.norm(A, axis=0)
    B_norm = np.linalg.norm(B, axis=0)

    D = np.divide((A * B).sum(axis=0), (A_norm * B_norm),
        out=np.zeros_like(A_norm), 
        where=(A_norm!=0) * (B_norm!=0)) # avoid divide by zero using where mask

    return 1 - D

# similarity functions --------------------------------------------------------

def spectral_similarity(y1, y2, mono=False, feature='magspec', metric='cosine', sr=44100, **kwargs):
    """Spectral similarity between two waveforms."""

    # which feature
    if feature == 'mfcc':
        extract_feature = extract_mfcc
    elif feature == 'magspec':
        extract_feature = extract_magspec
    elif feature == 'cqt':
        feature = extract_cqt

    # which metric
    metric = cosine_distance

    # resize to shortest of the two
    min_size = min(y1.shape[0], y2.shape[0])
    y1 = y1[:min_size]
    y2 = y2[:min_size]

    # extract features
    Y1 = extract_feature(y1, sr)
    Y2 = extract_feature(y2, sr)

    # distance
    D = 1 - metric(Y1, Y2)
    D_avg = np.average(D)

    return D_avg
