from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.md') as f:
    license = f.read()

requirements = [
    'numpy',
    'matplotlib',
    'pretty_midi',
    'pyFluidSynth >= 1.2.5',
    'librosa',
    'essentia',
    'keras',
    'tensorflow',
]

setup(
    name='mai',
    version='0.0.1',
    description='Music and Artificial Intelligence',
    long_description=readme,
    author='David Kant',
    author_email='dkant@ucsc.edu',
    url='https://canvas.ucsc.edu/courses/12767',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requirements
)
