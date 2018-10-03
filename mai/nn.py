import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np

# Network Architectures ----------------------------------------------------------------

def singleLayerPerceptron(units=2, input_dim=2, lr=0.03, decay=1e-6, momentum=0.9, nesterov=True):
    """A single layer perceptron."""

    # create an empty ANN
    model = Sequential()
    
    # add one layer
    model.add(Dense(units=units, input_dim=input_dim, activation='sigmoid' if units == 2 else 'softmax'))
    
    # compile the model
    model.compile(loss='binary_crossentropy' if units == 2 else 'categorical_crossentropy',
                  optimizer=SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov),
                  metrics=['accuracy'])
    
    return model


def multiLayerPerceptron(num_neurons=4, num_hidden_layers=1, input_dim=2, output_dim=2, 
                         activation='relu', lr=0.03, decay=1e-6, momentum=0.9, nesterov=True):
    """A mutlilayer perceptron."""

    # create an empty ANN
    model = Sequential()

    # add the first layer
    model.add(Dense(units=num_neurons, input_dim=input_dim, activation=activation))

    # add hidden layers
    for i in range(num_hidden_layers-1):
        model.add(Dense(units=num_neurons, activation=activation))
    
    # add the output layer
    model.add(Dense(units=output_dim, activation='sigmoid' if output_dim == 2 else 'softmax'))

    # compile the model
    model.compile(loss='binary_crossentropy' if output_dim == 2 else 'categorical_crossentropy',
                  optimizer=SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov),
                  metrics=['accuracy'])

    return model


# Training Data ------------------------------------------------------------------------

def training_data_platinum_records():
    """Training data from class."""
    
    # list of input/output pairs
    training_data = [
            ((0.07, 0.20), (1, 0)),
            ((0.19, 0.70), (1, 0)),
            ((0.30, 0.48), (1, 0)),
            ((0.45, 0.50), (1, 0)),
            ((0.53, 0.83), (1, 0)),
            ((0.70, 0.80), (1, 0)),
            ((0.91, 0.15), (1, 0)),
            ((0.33, 0.10), (0, 1)),
            ((0.55, 0.23), (0, 1)),
            ((0.62, 0.97), (0, 1)),
            ((0.72, 0.47), (0, 1)),
            ((0.77, 0.29), (0, 1)),
            ((0.90, 0.65), (0, 1)),
            ((0.95, 0.40), (0, 1)),
    ]

    # split into X and Y
    x_train = np.array([x for x,y in training_data])
    y_train = np.array([y for x,y in training_data])
    
    return x_train, y_train


def OR(x): 
    return [1,0] if (x[0] > 0.5) or (x[1] > 0.5) else [0,1]

def NOR(x): 
    return [1,0] if not(x[0] > 0.5 and x[1] > 0.5) else [0,1]

def XOR(x): 
    return [1,0] if (x[0] > 0.5) != (x[1] > 0.5) else [0,1]


def generate_training_data(num_samples, input_dim, func=lambda x: x):
    """Generate training data according to some function."""

    x_train = np.random.random((num_samples, input_dim))
    y_train = np.array([func(e) for e in x_train])
    
    return x_train, y_train


# Plots --------------------------------------------------------------------------------
import matplotlib.pyplot as plt

def prediction_plot(x_test, y_test, ax=None, alpha=0.5):

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(x_test[:,0], x_test[:,1], c=plt.cm.viridis(y_test.argmax(axis=1).astype(float)), alpha=alpha)
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    ax.set_title('Predictions')


def boundary_plot(model, x_train, y_train, ax=None, alpha=1):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        
    x,y = np.meshgrid(np.linspace(-0.06, 1.06, 300), np.linspace(-0.06, 1.06, 300))
    z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=300*300).argmax(axis=1).reshape((300,300))

    CS = ax.contour(x, y, z, colors='r', linewidths=1, alpha=0.2)
    ax.scatter(x_train[:,0], x_train[:,1], c=plt.cm.viridis(y_train.argmax(axis=1).astype(float)), alpha=alpha)
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    ax.set_title('Decision Boundary')
    

def prediction_and_boundary_plot(model, x_test, y_test, x_train, y_train):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    prediction_plot(x_test, y_test, ax=ax[0], alpha=0.5)
    boundary_plot(model, x_train, y_train, ax=ax[1], alpha=0.5)

    
def colormesh_plot(model, ax=None, alpha=1):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
            
    x,y = np.meshgrid(np.linspace(-0.06, 1.06, 300), np.linspace(-0.06, 1.06, 300))
    z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=300*300).argmax(axis=1).reshape((300,300))

    CM = ax.pcolormesh(x, y, z, cmap='viridis', alpha=0.5)
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    ax.set_title('Decision Boundary')


# Animations ---------------------------------------------------------------------------

from matplotlib import animation
from IPython.display import HTML

def prediction_animation(model, x_train, y_train, x_test, batch_size=20, frames=50, 
                         epochs_per_frame=1, fps=33, format='browser'):
    
    # figure setup
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    scat = plt.scatter(x_test[:,0], x_test[:,1], animated=True, alpha=0.5)
    plt.close()
    
    # update function
    def update(i, scat):
        if i == 0:
            loss, acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        else: 
            history = model.fit(x_train, y_train, epochs=epochs_per_frame, batch_size=batch_size, verbose=0)
            acc = history.history['acc'][0]
        y_test = model.predict(x_test, batch_size=len(x_test))
        scat.set_color(plt.cm.viridis(y_test.argmax(axis=1).astype(float)))
        ax.set_title('Epoch {0}, Accuracy {1:.2f}'.format(i*epochs_per_frame, acc))
        return scat,

    # animation
    anim = animation.FuncAnimation(fig, update, frames=frames, fargs=(scat,), interval=fps, blit=True)

    # run it as html
    if format == 'browser':
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # For google colab
        return HTML(anim.to_html5_video())

    # return animation
    if format == 'animation':
        return anim
        
        
def boundary_animation(model, x_train, y_train, x_test, batch_size=20, frames=50, 
                       epochs_per_frame=1, fps=33, format='browser'):
    
    # figure setup
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    x,y = np.meshgrid(np.linspace(-0.06, 1.06, 100), np.linspace(-0.06, 1.06, 100))
    z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=100*100).argmax(axis=1).reshape((100,100))  
    cax = ax.contour(x, y, z, colors='r', linewidths=1, alpha=0.2, animated=True)
    sax = ax.scatter(x_train[:,0], x_train[:,1], c=plt.cm.viridis(y_train.argmax(axis=1).astype(float)), alpha=0.5, animated=True)
    plt.close()
        
    # update function
    def update(i, ax):
        if i == 0:
            loss, acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        else: 
            history = model.fit(x_train, y_train, epochs=epochs_per_frame, batch_size=batch_size, verbose=0)
            acc = history.history['acc'][0]
        x,y = np.meshgrid(np.linspace(-0.06, 1.06, 100), np.linspace(-0.06, 1.06, 100))
        z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=100*100).argmax(axis=1).reshape((100,100)) 
        ax.clear()
        ax.contour(x, y, z, colors='r', linewidths=1, alpha=0.2)
        ax.scatter(x_train[:,0], x_train[:,1], c=plt.cm.viridis(y_train.argmax(axis=1).astype(float)), alpha=0.5)
        ax.set_title('Epoch {0}, Accuracy {1:.2f}'.format(i*epochs_per_frame, acc))
        return ax,

    # animation
    anim = animation.FuncAnimation(fig, update, frames=frames, fargs=(ax,), interval=fps, blit=False)

    # run it as html
    if format == 'browser':
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # For google colab
        return HTML(anim.to_html5_video())

    # return animation
    if format == 'animation':
        return anim
    

def prediction_and_boundary_animation(model, x_train, y_train, x_test, batch_size=20, frames=50, 
                                      epochs_per_frame=1, fps=33, format='browser'):
        
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
    # setup scatter
    ax[0].set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    scat = ax[0].scatter(x_test[:,0], x_test[:,1], animated=True, alpha=0.5)

    # setup contour  
    ax[1].set(xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    x,y = np.meshgrid(np.linspace(-0.06, 1.06, 100), np.linspace(-0.06, 1.06, 100))
    z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=100*100).argmax(axis=1).reshape((100,100))
    cax = ax[1].contour(x, y, z, colors='r', linewidths=1, alpha=0.2, animated=True)
    sax = ax[1].scatter(x_train[:,0], x_train[:,1], c=plt.cm.viridis(y_train.argmax(axis=1).astype(float)), alpha=0.5, animated=True)
    
    plt.close()
    
    # update function
    def update_scatter(i, scat, acc):
        y_test = model.predict(x_test, batch_size=len(x_test))
        scat.set_color(plt.cm.viridis(y_test.argmax(axis=1).astype(float)))
        ax[0].set_title('Predictions')
        ax[0].set_xlabel('Epoch {0}, Accuracy {1:.2f}'.format(i, acc))
    
    # update function
    def update_contour(i, ax, acc):
        x,y = np.meshgrid(np.linspace(-0.06, 1.06, 100), np.linspace(-0.06, 1.06, 100))
        z = model.predict(np.array([x.flatten(), y.flatten()]).T, batch_size=100*100).argmax(axis=1).reshape((100,100)) 
        ax.clear()
        ax.contour(x, y, z, colors='r', linewidths=1, alpha=0.2)
        ax.scatter(x_train[:,0], x_train[:,1], c=plt.cm.viridis(y_train.argmax(axis=1).astype(float)), alpha=0.5)
        ax.set_title('Decision Boundary')
        ax.set_xlabel('Epoch {0}, Accuracy {1:.2f}'.format(i, acc))
    
    # update function
    def update(i, scat, ax):
        if i == 0: 
            loss, acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
        else: 
            history = model.fit(x_train, y_train, epochs=epochs_per_frame, batch_size=batch_size, verbose=0)
            acc = history.history['acc'][0]
        update_scatter(i*epochs_per_frame, scat, acc)
        update_contour(i*epochs_per_frame, ax[1], acc)
        return ax,
    
    # animation
    anim = animation.FuncAnimation(fig, update, frames=frames, fargs=(scat, ax,), interval=fps, repeat=True, blit=False)

    # run it as html
    if format == 'browser':
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # For google colab
        return HTML(anim.to_html5_video())

    # return animation
    if format == 'animation':
        return anim
