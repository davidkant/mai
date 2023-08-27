# Music and Artificial Intelligence

This code repository is the home of the [Music AI Tutorial](http://davidkantportfolio.com/music-ai-tutorial/), a collection of cloud-based, interactive tutorials that teach topics in music, artificial intelligence, and computer science. The tutorial takes a historical approach. My intention is to give cultural, technical, and aesthetic context to contemporary approaches to Music AI.

### Viewing the tutorials

The tutorials are intended to be viewed using [Google Colaboratory](https://drive.google.com/drive/folders/11KtF1-QpE-qKRA2LWVIf2bdA_a9Afn49), a cloud-based Jupyter Notebook environment, which allows you to edit and run code in the browser. You can also view a static render of the tutorials via [nbviewer](https://nbviewer.jupyter.org/github/davidkant/mai/tree/master/tutorial/), or an interactive versions via [MyBinder](https://mybinder.org/v2/gh/davidkant/mai/master). 

### Teaching

The ***Music AI Tutorial*** was developed for the course ***MUSC 80L: Artificial Intelligence and Music*** at UC Santa Cruz Course with instructor David Kant. I am actively seeking new opportunities to teach this course. Please contact me if you are interested!

### Thanks!
This project is in ongoing development, so please check back for updates!

### Local Installation

If you want to use the mai package locally (not in Collab/Jupyter notebook), you can install it with pip:

```bash
python3 -m pip install git+https://github.com/davidkant/mai
# or
git clone https://github.com/davidkant/mai
python3 -m pip install ./mai
```

The `tensorflow`/`keras` dependencies could be tricky to install sometimes, [see docs](https://www.tensorflow.org/install/pip) if you're having issues.

Then, you can import it like any other package. Without setting up the proper libraries for audio playback, it might be easiest to save to `MIDI`.

```python
import mai

midi = mai.make_music_heterophonic(
    [[35, 35], [35, 38, 38], [42, 42, 42, 42, 42, 42, 42, 42, 49]],
    durs=[[1.3, 1.3], [0.65, 1.3, 1.3], [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 1.95]],
    is_drum=True,
    format="MIDI"
)

midi.write("my_music.mid")
```

Then you can visualize it with an online tool or with a tool like [fluidsynth](https://www.fluidsynth.org/) or [timidity](https://timidity.sourceforge.net/). Note these typically require you to setup a [soundfont](https://en.wikipedia.org/wiki/SoundFont)
