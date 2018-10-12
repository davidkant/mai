import librosa
import os 
import IPython.display


def play_mp3(x, mp3=True, sr=22050):
    """Write audio to mp3 and return Ipython widget."""
    # write temp wav file
    librosa.output.write_wav('temp.wav', x, sr=sr)
    # convert to mp3 using lame
    os.system("lame temp.wav temp.mp3 --quiet")
    # return ipython audio widget
    return IPython.display.Audio('temp.mp3')

def write_mp3(x, filename='temp.mp3', sr=22050):
    """Write audio to mp3."""
    # write temp wav file
    librosa.output.write_wav('temp.wav', x, sr=sr)
    # convert wav file to mp3 using lame
    os.system("lame temp.wav {} --quiet".format(filename))
