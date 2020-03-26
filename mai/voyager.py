from .music_makers import make_music
from .music_makers import make_music_heterophonic
from .music_makers import make_music_plot
from .music_makers import make_music_heterophonic_plot
from .listen import ppitch
from .musifuncs import hz_to_midi

from functools import reduce
import random
import numpy as np
import matplotlib.pyplot as plt

from librosa.core import resample
import IPython.display
from IPython.display import display, HTML


"""
    # TODO:
    - override pgm=1
    - use return statements
    - edit doc to reflect storing instance variables
    - args and doc are burried down one layer on many funcs including audio to midi
    - write a to string function to display parameters
"""


FEATURES = [
    "pitch_center",
    "pitch_range",
    "step_size",
    "contour",
    "microtonal",
    "pulse",
    "rhythm",
    "length",
]

FEATURES_DISPLAY_ORDER = [
    "pitch_center",
    "pitch_range",
    "step_size",
    "contour",
    "microtonal",
    "pulse",
    "rhythm",
    "length",
]

FEATURES_DISPLAY_STRINGS = {
    "pitch_center": "Pitch Center",
    "pitch_range": "Pitch Range",
    "step_size": "Step Size",
    "contour": "Contour",
    "microtonal": "Microtonal",
    "pulse": "Pulse",
    "rhythm": "Rhythm",
    "length": "Num Notes",
}


class Voyager:
    """A toy version of George Lewis' piece Voyager."""

    def __init__(self):
        self.listener = Listener()
        self.orchestra = Orchestra()

    def listen(self, y):
        """Listen."""
        self.listener.audio_to_midi(y)
        self.listener.extract_musical_features()

    def plot_audio_to_midi(self):
        """Plot audio to midi transcription."""
        self.listener.plot_audio_to_midi()

    def play_audio_to_midi(self, pgm=1):
        """Play audio to midi transcription."""
        self.listener.play_audio_to_midi(pgm=pgm)

    def print_musical_features(self):
        """Print listener analysis features."""
        self.listener.print_musical_features()

    def plot_musical_features(self):
        """Plot listener analysis features."""
        self.listener.plot_musical_features()
        self.listener.plot_scale_and_pitch_probs()

    def setphrasebehavior(self, kind='manual', params=None):
        """Set response parameters.

        Optional kwarg to force a response_behavior:
          - one of {'manual', 'ignore', 'imitate', 'oppose'}
          - 'manual' uses the params kwarg to pass params
        """
        if kind is 'ignore':
            self.set_ignore_response()
        elif kind is 'imitate':
            self.set_imitate_response()
        elif kind is 'oppose':
            self.set_oppose_response()
        else:
            self.set_manual_response(**params)

    def respond(self):
        """Generate a response using current parameter values."""
        self.orchestra.response()

    def set_ignore_response(self):
        """GenSeterate an ignore response parameters."""
        self.orchestra.ignore_response()

    def set_imitate_response(self, num_players=None):
        """Generate an imitate response parameters."""
        self.orchestra.imitate_response(self.listener.features, num_players)

    def set_oppose_response(self):
        """Generate an oppose response parameters."""
        self.orchestra.oppose_response(self.listener.features)

    def set_manual_response(self, **params):
        """Generates a manual response parameters."""
        self.orchestra.manual_response(**params)

    def plot_response(self):
        """Plot orchestra response."""
        self.orchestra.plot_response()

    def play_response(self, sr=16000, format='inbrowser'):
        """Play orchestra response."""
        self.orchestra.play_response(sr=sr, format=format)

    def print_response(self):
        """Print BOTH orchestra and listener features."""
        self.orchestra.print_response(self.listener.features)

    def play_together(self, sr_y=44100, sr_z=16000):
        """Render y and response to audio together."""
        self.orchestra.play_together(self.listener.y, sr_y=sr_y, sr_z=sr_z)


class Listener:
    """The listener listens."""

    def __init__(self):
        self.y = None
        self.pitches = None
        self.durations = None
        self.features = None

    def audio_to_midi(
        self,
        y,
        pitch_thresh=0.3,
        width=3,
        hold_time=4,
        vibrato_time=3,
        confidence_thresh=0.1,
        last_dur=0.2,
        hop_length=512,
        n_fft=2048,
        win_length=2048,
        sr=44100,
    ):
        """Audio to MIDI transcription.

        Note: this is pretty bare bones. There is much to add...

        TODO:
            - onsets are triggered by change in pitch only -> look for stabilities!
            - vibrato time debounce short-lived changes
            - offset detection, especially for the last note!
            - make polyphonic

        Args
            y: Audio file to transcribe.
            pitch_thresh: Pitch onset threshold.
            width: Change in pitch is measured within a window.
            hold_time: Momentarily suppress onsets after an onset is found.
            confidence_thresh: Suppress onsets with low confidence.
            last_dur: Duration of last onset.
            hop_length: FFT hop.
            n_fft: FFT size.
            win_length: FFT window size.

        Returns
            pitches: List of pitches.
            durations: List of durations.
        """
        # pitch estimation
        pitches, h, D, peaks, confidences, t = ppitch(
            y,
            sr=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            num_peaks=10,
            num_pitches=1,
            min_fund=60,
            max_fund=2000,
            max_peak=20000,
            npartial=4,
            ml_width=50,
            bounce_width=50,
        )

        # convert frequency in Hz to MIDI number
        pitches = hz_to_midi(pitches)

        # zero out
        onsets = []
        holding = False

        # loop
        for i in range(width, len(pitches) - width):
            if holding > 0:
                holding -= 1
                continue
            if confidences[i] < confidence_thresh:
                continue
            if (
                abs(max(pitches[i - width : i]) - min(pitches[i - width : i]))
                >= pitch_thresh
            ):
                onsets.append(i)
                holding = hold_time

        # reformat data
        onsets = np.array(onsets)
        pitches_cooked = pitches[onsets]
        pitches_cooked = pitches_cooked[:, 0]
        durs = np.diff(onsets) * hop_length / sr
        durs = np.concatenate((durs, [last_dur]))

        self.y = y
        self.pitches = pitches_cooked
        self.durations = durs

    def extract_musical_features(self, agg='median'):
        """Extract musical features.

        Args:
            pitches: list of pitches.
            durations: list of durations.
            agg: aggregate function {'media', 'mean'}.

        Returns:
            A dict containing musical features.
        """
        pitches = self.pitches
        durations = self.durations

        pitches_12tet = [int(round(x)) for x in pitches]
        pitches_mod12 = [int(round(x)) % 12 for x in pitches]

        features = dict()
        features["scale"] = sorted(set(pitches_12tet))
        features["pitch_set"] = sorted(set(pitches_mod12))
        features["pitch_probs"] = [pitches_12tet.count(x) for x in set(pitches_12tet)]
        features["pitch_center"] = int(round(np.average(pitches)))
        features["pitch_range"] = 48
        if agg is 'mean':
            features["step_size"] = np.average(np.abs(np.diff(pitches)))
            features["contour"] = np.average(np.sign(np.diff(pitches)))
            features["microtonal"] = np.average(np.abs(pitches - pitches_12tet))
            features["pulse"] = 1000.0 / (np.average(durations) * 1000.0) * 60 * 0.5
        if agg is 'median':
            features["step_size"] = np.median(np.abs(np.diff(pitches)))
            features["contour"] = np.average(np.sign(np.diff(pitches)))
            features["microtonal"] = np.median(np.abs(pitches - pitches_12tet))
            features["pulse"] = 1000.0 / (np.median(durations) * 1000.0) * 60 * 0.5
        features["length"] = int(len(pitches))
        features["rhythm"] = np.var(durations)

        self.features = features

    def plot_audio_to_midi(self):
        """Plot audio to midi transcription."""
        return make_music_plot(list(self.pitches), durs=list(self.durations))

    def play_audio_to_midi(self, pgm=1):
        """Play audio to midi transcription."""
        d = make_music(
            list(self.pitches),
            durs=list(self.durations),
            pgm=pgm,
            format='inbrowser'
        )
        return IPython.display.display(d)

    def print_musical_features(self):
        """Pretty print musical features data."""

        for i, feature_name in enumerate(FEATURES_DISPLAY_ORDER):
            print("{0}. {1}: {2:.2f}".format(
                i+1,
                FEATURES_DISPLAY_STRINGS[feature_name],
                self.features[feature_name]
            ))

    def plot_musical_features(self):
        """Pretty plot musical features data."""

        FEATURES_RANGES = {
            "pitch_center": ([30, 90], "linear"),
            "pitch_range": ([0, 60], "linear"),
            "step_size": ([0, 12], "linear"),
            "contour": ([-1, 1], "linear"),
            "microtonal": ([0, 1], "linear"),
            "pulse": ([20, 1000], "log"),
            "rhythm": ([0.01, 1], "log"),
            "length": ([0, 100], "linear"),
        }

        fig, ax = plt.subplots(1, len(FEATURES_DISPLAY_ORDER), figsize=(8, 2.6))

        for i, f in enumerate(FEATURES_DISPLAY_ORDER):
            ax[i].bar(
                0.5,
                self.features[f],
                tick_label=FEATURES_DISPLAY_STRINGS[f],
                width=0.8,
                align="center",
                color="C1",
                lw=2,
                alpha=0.666,
            )
            ax[i].set_ylim(FEATURES_RANGES[f][0])
            ax[i].set_yscale(FEATURES_RANGES[f][1])
            ax[i].set_xlim([0, 1])

        plt.tight_layout()
        plt.show()

    def plot_scale(self, ylim=[30, 90], ax=None, show=False):
        """Display pitch set as a bar graph."""

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        ax.bar(
            range(len(self.features["scale"])),
            self.features["scale"],
            capsize=5,
            fc="C4",
            color="b",
        )

        ax.set_ylim(ylim)
        ax.set_xticks([])

        ax.set_title("Scale")
        ax.set_ylabel("Pitch (MIDI #)")

        if show:
            plt.show()

    def plot_pitch_probs(self, ax=None, show=False):
        """Plot pitch probability distribution."""

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        ax.plot(self.features["scale"], self.features["pitch_probs"], lw=3, c="C4")

        ax.fill_between(
            self.features["scale"],
            [0] * len(self.features["scale"]),
            self.features["pitch_probs"],
            color="C4",
            alpha=0.5,
        )

        ax.set_title("Pitch Probabilities")
        ax.set_xlabel("Pitch (MIDI #)")
        ax.set_ylabel("Weight / Probability")

        if show:
            plt.show()

    def plot_scale_and_pitch_probs(self):
        """Convenience to display both."""

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        self.plot_scale(ax=ax[0])
        self.plot_pitch_probs(ax=ax[1])

        plt.tight_layout()
        plt.show()


class Orchestra:
    """The orchestra plays."""

    def __init__(self):
        self.pitches = None
        self.durations = None
        self.parameters = None
        self.response_type = 'ignore'

    # TODO: function for now, could be a class later...
    def player(
        self,
        pitch_center=60,
        pitch_range=12,
        step_size=6,
        step_weight=1.0,
        contour=-0.5,
        microtonal=0.0,
        scale=range(30, 90),
        scale_weight=0.5,
        pitch_set=[0, 2, 4, 6, 8, 10],
        pitch_set_weight=1.0,
        pitch_probs=[1 for i in range(30, 90)],
        pulse=120,
        rhythm=0.5,
        length=33,
        volume=100,
    ):
        """Simple version of a Player for George Lewis' Voyager.

        This is a probabilistic model: ``never say never''.

        TODO:
            - pitch probs not yet implemented

        Would be cool to add and estimate from data on listener:
            - repeat_prob
            - repeat_memory

        Note: if no next pitches are possible, use previous pitch.

        Args:
            pitch_center: Center of pitch range.
            pitch_range: Min/max of pitch range from center.
            step_size: Max step size.
            step_weight: How much to prioritize within step_size.
            contour: Likelihood to move up/down direction.
            microtonal: Microtonal inflection.
            scale: Restrict to these pitches.
            scale_weight: How much to prioritize within scale.
            pitch_set: Restrict to these pitch classes.
            pitch_set_weight: How much to prioritize within pitch_set.
            pitch_probs: Pitch set probabilities.
            pulse: Average duration.
            rhythm: Deviation from the pulse.
            length: Number of notes to generate.
            volume: Note velocity.

        Return:
            music:
        """

        # start with empty lists for both pitch and duration
        my_pitches = []
        my_durs = []

        # loop until we have enough notes
        while len(my_durs) < length:

            # first pitch
            if len(my_durs) < 1:
                new_pitch = random.randint(
                    pitch_center - pitch_range, pitch_center + pitch_range
                )

            # not first pitch
            else:
                prev_pitch = my_pitches[-1]
                alpha = range(90)

                w_range = np.array([
                    (x >= pitch_center - pitch_range)
                    and (x <= pitch_center + pitch_range)
                    for x in alpha
                ])

                w_step = np.array([
                    1
                    if (x >= prev_pitch - step_size)
                    and (x <= prev_pitch + step_size)
                    else 1 - step_weight
                    for x in alpha
                ])

                w_contour = np.array([
                    contour - -1
                    if (x - prev_pitch) > 0
                    else 1 - contour
                    for x in alpha
                ])

                w_pitchset = np.array([
                    1
                    if x % 12 in pitch_set
                    else 1 - pitch_set_weight
                    for x in alpha
                ])

                w_scale = np.array([
                    1
                    if x in scale
                    else 1 - scale_weight
                    for x in alpha
                ])

                w_all = w_range * w_step * w_contour * w_pitchset

                if np.sum(w_all) == 0.0:
                    new_pitch = prev_pitch
                else:
                    new_pitch = random.choices(alpha, weights=w_all)[0]

            # microtonal pitch adjustment
            new_pitch += random.uniform(-microtonal, microtonal)

            # choose duration
            new_dur = (60.0 / pulse / 2) * random.uniform(1 - rhythm, 1 + rhythm)

            # append to the melody
            my_pitches += [new_pitch]
            my_durs += [new_dur]

        return my_pitches, my_durs

    def response(self):
        """Generates an orchestra response with the current self.parameters.

        Returns:
            self.pitches: Lists of pitches.
            self.durations: Lists of durations.
        """
        # call players
        # TODO: write function orch_params to player_params
        my_music = [self.player(
                pitch_center=self.parameters['pitch_center'],
                pitch_range=self.parameters['pitch_range'],
                step_size=self.parameters['step_size'],
                step_weight=self.parameters['step_weight'],
                contour=self.parameters['contour'],
                microtonal=self.parameters['microtonal'],
                scale=self.parameters['scale'],
                scale_weight=self.parameters['scale_weight'],
                pitch_set=self.parameters['pitch_set'],
                pitch_set_weight=self.parameters['pitch_set_weight'],
                pitch_probs=self.parameters['pitch_probs'],
                pulse=self.parameters['pulse'],
                rhythm=self.parameters['rhythm'],
                length=self.parameters['length'],
                volume=self.parameters['volume'],
            ) for i in range(self.parameters['num_players'])]

        # combine multiple voices
        pitches = [x[0] for x in my_music]
        durs = [x[1] for x in my_music]

        # store as member variables
        self.pitches = pitches
        self.durations = durs

    def ignore_response(self):
        """Ignore listener data and generate a random response.

        Note:
          - length is randomized here.

        Returns:
            response parameters: parameter dict.
        """
        response_parameters = self.random_parameters()
        self.parameters = response_parameters

    def imitate_response(self, listener_features, num_players=None):
        """Generate a response that imitates listener data.

        Note:
          - length is imitated here.

        Returns:
            response parameters: parameter dict.
        """
        # random parameters
        response_parameters = self.random_parameters()

        # swap in for imitation
        for param in listener_features.keys():
            response_parameters[param] = listener_features[param]

        # force one player
        if num_players:
            response_parameters["num_players"] = num_players

        self.parameters = response_parameters

    def oppose_response(self, listener_features):
        """Generate a response that opposes listener data.

        Note: length is opposd here.

        Returns:
            response parameters: parameter dict.
        """

        FEATURES_RANGES = {
            "pitch_center": ([30, 90], "linear"),
            "step_size": ([0, 12], "linear"),
            "contour": ([-1, 1], "linear"),
            "microtonal": ([0, 1], "linear"),
            "pulse": ([20, 1000], "log"),
            "rhythm": ([0.01, 1], "log"),
        }

        OPPOSE = [
            # 'num_players',
            'pitch_center',
            'pitch_range',
            'step_size',
            'step_weight',
            'contour',
            'microtonal',
            'scale',
            'scale_weight',
            'pitch_set',
            'pitch_set_weight',
            'pitch_probs',
            'pulse',
            'rhythm',
            'length',
            'volume',
        ]

        response_parameters = self.random_parameters()

        response_parameters['pitch_center'] = ((listener_features['pitch_center'] + 60 / 2) -30) % 60 + 30
        response_parameters['pitch_range'] = (listener_features['pitch_range'] + 48 / 2) % 48
        response_parameters['step_size'] = (listener_features['step_size'] + 12 / 2) % 12
        response_parameters['contour'] = listener_features['contour'] * -1
        response_parameters['microtonal'] = (listener_features['microtonal'] + 1.0 / 2) % 1
        # response_parameters['pulse'] = (listener_features['pulse'] + 500 / 2) % 450
        response_parameters['rhythm'] = (listener_features['rhythm'] + 1.0 / 2) % 1
        response_parameters['length'] = (listener_features['length'] + 66 / 2) % 66

        # response_parameters['pitch_center'] = 90 - listener_features['pitch_center'] + 30
        # response_parameters['pitch_range'] = 48 - listener_features['pitch_range']
        # response_parameters['step_size'] = 12 - listener_features['step_size']
        # response_parameters['contour'] = listener_features['contour'] * -1
        # response_parameters['microtonal'] = 1 - listener_features['microtonal']
        response_parameters['pulse'] = (200 - listener_features['pulse']) % 200
        # response_parameters['rhythm'] = 1 - listener_features['rhythm']

        # scale tbd
        # pitch set tbd

        # but keep length the same!
        # response_parameters["length"] = listener_features["length"]

        self.parameters = response_parameters

    def manual_response(self, **params):
        """Generate a manual response using given parameters."""

        response_parameters = self.default_parameters();

        for k,v in params.items():
            if k in response_parameters:
                response_parameters[k] = v

        self.parameters = response_parameters

    def default_parameters(self):
        """Return default response parameters.

        These are overriden by kwargs in call to self.response(), so these really
        just serve to make sure no parameters are omitted.
        """

        response_parameters = {
            'num_players': 5,
            'pitch_center': 60,
            'pitch_range': 12,
            'step_size': 6,
            'step_weight': 1.0,
            'contour': -0.5,
            'microtonal': 0.0,
            'scale': range(30, 90),
            'scale_weight': 0.5,
            'pitch_set': [0, 2, 4, 6, 8, 10],
            'pitch_set_weight': 1.0,
            'pitch_probs': [1 for i in range(30, 90)],
            'pulse': 120,
            'rhythm': 0.5,
            'length': 33,
            'volume': 100,
            'pgm': 1,
            'is_drum': False,
        }
        return response_parameters

    # TODO: should start with call to default_parameters() so don't miss any...
    # TODO: and these should live in a spec
    def random_parameters(self):
        """Return random response parameters."""

        scale_length = random.randint(6, 36)
        pitch_set_length = 9

        random_parameters = {
            'num_players': int(round(random.triangular(1, 9, 2))),
            'pitch_center': random.randint(30, 90),
            'pitch_range': random.randint(1, 12),
            'step_size': random.randint(1, 12),
            'step_weight': random.uniform(0.9, 1),
            'contour': random.uniform(-1, 1),
            'microtonal': random.uniform(-1, 1),
            'scale': sorted([random.randint(30, 90) for i in range(scale_length)]),
            'scale_weight': random.uniform(0.9, 1),
            'pitch_set': set([random.randint(0, 11) for i in range(pitch_set_length)]),
            'pitch_set_weight': 0.5,
            'pitch_probs': [random.randint(0, 12) for x in range(scale_length)],
            'pulse': random.triangular(2, 200, 30),
            'rhythm': random.uniform(0, 0.5),
            'length': random.randint(1, 66),
            'volume': 100,
            'pgm': int(random.randint(1, 120)),
            'is_drum': False,
        }
        return random_parameters

    def play_response(self, sr=16000, format='inbrowser'):
        """Render to audio and play."""

        d = make_music_heterophonic(
            self.pitches,
            durs=self.durations,
            pgm=self.parameters['pgm'],
            is_drum=self.parameters['is_drum'],
            format=format,
            sr=sr
        )

        if format=='inbrowser':
            return IPython.display.display(d)
        else:
            return d

    def plot_response(self):
        """Plot response as heterophonic music plot."""
        return make_music_heterophonic_plot(self.pitches, self.durations)

    def print_response(self, listener_features):
        """Display a table comparing features."""

        def format_header():
            # use <table style="width:50%"> to adjust size
            return ['<html><body><font size="2" face="menlo"><table border="0">']

        def format_first_row():
            return [
                "<tr><th>Feature</th>\
                <th>Listener</th>\
                <th></th\
                ><th>Orchestra</th></tr>"
            ]

        def format_row(feature, listener_val, relation, player_val):
            return '<tr>\
                    <td>{0}:</td>\
                    <td align="right">{1:.2f}</td>\
                    <td>{2}</td>\
                    <td>{3:.2f}</td>\
                    </tr>'.format(
                        feature, listener_val, relation, player_val,
                    )

        def format_footer():
            return ["</table></font></body></html>"]

        header = format_header()
        first_row = format_first_row()
        rows = [
            format_row(
                FEATURES_DISPLAY_STRINGS[k],
                listener_features[k],
                "&nbsp;-->&nbsp;",
                self.parameters[k],
            )
            for k in FEATURES_DISPLAY_ORDER
        ]
        footer = format_footer()

        table = reduce(lambda x, y: x + y, header + first_row + rows + footer)
        return display(HTML(table))

    def play_together(self, y, sr_y=44100, sr_z=16000):
        """Mix together input and response audio and play back.

        Note:
          - resample original audio y to 16000 to match MIDI resynth
          - pans hard left and hard right b/c why not
        """

        z = self.play_response(format='audio', sr=sr_z)

        y_rs = resample(y, sr_y, sr_z)

        if len(y_rs) < len(z):
            y_hat = np.pad(y_rs, (0, len(z) - len(y_rs)))
            mix = np.vstack((y_hat, z))

        if len(z) < len(y_rs):
            z_hat = np.pad(z, (0, len(y_rs) - len(z)))
            mix = np.vstack((y_rs, z_hat))

        d = IPython.display.Audio(mix, rate=sr_z)
        IPython.display.display(d)
