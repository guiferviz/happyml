
"""
Plotting module.
"""

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.colors import rgb2hex
from matplotlib.colors import LinearSegmentedColormap

import happyml


rgb_colors = {
    "set1" : np.array([
        (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
        (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
        (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
        (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
        (1.0, 0.4980392156862745, 0.0),
        (1.0, 1.0, 0.2),
        (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
        (0.9686274509803922, 0.5058823529411764, 0.7490196078431373),
        (0.6, 0.6, 0.6),
        (0.0, 0.0, 0.0)]),
    "set2" : np.array([]),
}

cdict  = {'red':   ((0.0, 1.0, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
                   
         #'alpha': ((0.0, 0.5, 0.5),
         #          (1.0, 0.5, 0.5))
        }
blue_red = LinearSegmentedColormap('BlueRed', cdict)
plt.register_cmap(cmap=blue_red)


def get_theme(prop=None):
    name_theme = happyml.config["theme"]

    if prop:
        return happyml.config["themes"][name_theme][prop]

    return happyml.config["themes"][name_theme]


def get_class_color(n_class, format="hex"):
    """Return the color of the spedified class.

    This function lookup the theme colors table and
    returns the color of the selected class.

    Args:
        n_class (number): Number from 1 to 10 or -1.
        format (string): hex or rgb.

    Return:
        a string if format = "hex" or rgb-tuple if
        format = "rgb".

    Raise:
        ValueError if you provide and unknow format.

    """
    hex_color = get_theme("colors")[n_class]

    if format.lower() == "hex":
        return hex_color
    elif format.lower() == "rgb":
        return hex2color(hex_color)
    else:
        ValueError("Unknow format, use 'hex' or 'rgb'")


def light_hex_color(color_hex, light=0.2):
    rgb = hex2color(color_hex)
    rgb = light_rgb_color(rgb, light=light)
    return rgb2hex(rgb)


def light_rgb_color(color, light=0.2):
    rgb = [i + light if i + light <= 1 else 1 for i in color]
    return tuple(rgb)


binary_ones_colors = (
    light_hex_color(get_class_color(0), light=0.3),
    light_hex_color(get_class_color(1), light=0.3),
)

binary_margin_colors = (
    light_hex_color(get_class_color(0), light=0.25),
    light_hex_color(get_class_color(0), light=0.35),
    light_hex_color(get_class_color(1), light=0.35),
    light_hex_color(get_class_color(1), light=0.25),
)


def predict_area(model, limits=None, samples=50,
                 x_samples=None, y_samples=None, **kwargs):
    """Evaluate a model on a rectangular area.

    Args:
        model (:attr:`happyml.models.Hypothesis`): Model to evaluate.
        limits (list): Position and dimension of the prediction area.
            Indicated by a list of the form [xmin, xmax, ymin, ymax].
            Defaults to ``[-1, 1, -1, 1]``.
        samples (int): number of x and y samples to be used. The more
            samples, the more precision. Defaults to 50.
        x_samples (int): Use it when you want a different number
            of samples for x axis. Defaults to ``samples``.
        y_samples (int): Use it when you want a different number
            of samples for y axis. Defaults to ``samples``.

    Returns:
        X, Y, Z (numpy.ndarray): 3 matrices of the same size, i.e. of size
        ``(y_samples, x_samples)``.

        **X** (numpy.ndarray): :math:`X_{ij}` contains the x coordinate used\
            to compute the :math:`Z_{ij}` value. All the rows in a column\
            contains the same number.

        **Y** (numpy.ndarray): :math:`Y_{ij}` contains the y coordinate used\
            to compute the :math:`Z_{ij}` value. All the columns in a row\
            contains the same number.

        **Z** (numpy.ndarray): result of appliying ``f`` on\
            :math:`(X_{ij}, Y_{ij})` coordinates. In other words\
            ``Z[i, j] = f(X[i, j], Y[i, j])``.

    Raises:
        ValueError: if `len` of `limits` is not 4.

    Example:
        .. code-block:: python

            def fun(X, Y):
                # Linear function. X and Y are matrices, then the output
                # will be a matrix (Z matrix).
                return X * w1 + Y * w2 + b

            # Evaluates function fun on a square centered on the origin
            # with sides of length 2 (from -1 to 1).
            X, Y, Z = grid_function(fun)
    """
    # Check parameters.
    if not limits:
        limits = [-1, 1, -1, 1]
    elif len(limits) != 4:
        raise ValueError("limits need 4 values: [xmin, xmax, ymin, ymax]")
    # Create linspaces and matrices with the x and y coordinates.
    x = np.linspace(limits[0], limits[1], num=x_samples or samples)
    y = np.linspace(limits[2], limits[3], num=y_samples or samples)
    X, Y = np.meshgrid(x, y)
    # Transforms the grids values to a matrix with two columns:
    # the first are the x coordinates, the second are the y coordinates.
    coordinates = np.array([X, Y]).reshape(2, -1).T
    # Use the model to predict an output on each coordinate pair.
    predicted = model.predict(coordinates)
    # Convert the predicted values to a matrix form.
    Z = predicted.reshape(X.shape)  # or Y.shape
    # Return all the usefull data.
    return X, Y, Z

def grid_function(f, bounds=[-1, 1, -1, 1], samples=50,
                  x_samples=None, y_samples=None):
    """Evaluate a function ``f`` on a rectangular area.

    Args:
        f (function): Function to be applied on each sample of the rectangular
            area. The function must receive two matrix by parameter: one with
            x coordinates an another with y coordinates.
        bounds (list): Position and dimension of the rectangular area.
            Indicated by a list of the form [xmin, xmax, ymin, ymax].
            Defaults to ``[-1, 1, -1, 1]``.
        samples (int): number of x and y samples to be used. The more
            samples, the more precision. Defaults to 50.
        x_samples (int): Use it when you want a different number
            of samples for x axis. Defaults to ``samples``.
        y_samples (int): Use it when you want a different number
            of samples for y axis. Defaults to ``samples``.

    Returns:
        X, Y, Z (numpy.ndarray): 3 matrices of the same size, i.e. of size
        ``(y_samples, x_samples)``.

        **X** (numpy.ndarray): :math:`X_{ij}` contains the x coordinate used\
            to compute the :math:`Z_{ij}` value. All the rows in a column\
            contains the same number.

        **Y** (numpy.ndarray): :math:`Y_{ij}` contains the y coordinate used\
            to compute the :math:`Z_{ij}` value. All the columns in a row\
            contains the same number.

        **Z** (numpy.ndarray): result of appliying `f` on\
            :math:`(X_{ij}, Y_{ij})` coordinates. In other words\
            ``Z[i, j] = f(X[i, j], Y[i, j])``.

    Raises:
        ValueError: if ``len`` of ``bounds`` is not 4.

    Example:
        .. code-block:: python

            def fun(X, Y):
                # Linear function. X and Y are matrices, then the output
                # will be a matrix (Z matrix).
                return X * w1 + Y * w2 + b

            # Evaluates function fun on a square centered on the origin
            # with sides of length 2 (from -1 to 1).
            X, Y, Z = grid_function(fun)

    See Also:
        :attr:`happyml.plot.grid_function_slow`

    """
    if len(bounds) != 4:
        raise ValueError("bounds need 4 values: [xmin, xmax, ymin, ymax]")
    x = np.linspace(bounds[0], bounds[1], num=x_samples or samples)
    y = np.linspace(bounds[2], bounds[3], num=y_samples or samples)
    X, Y = np.meshgrid(x, y)
    return X, Y, f(X, Y)


def grid_function_slow(f, bounds=[-1, 1, -1, 1], samples=50,
                       x_samples=None, y_samples=None):
    """Evaluate a function ``f`` on a rectangular area.

    This function is slow because the function `f` does not receive
    matrices, only 2 numbers.

    See Also:
        :attr:`happyml.plot.grid_function`

    """
    bounds = [-1, 1, -1, 1] if bounds is None else bounds
    if len(bounds) != 4:
        raise ValueError("bounds need 4 values: [xmin, xmax, ymin, ymax]")

    x = np.linspace(bounds[0], bounds[1], num=x_samples or samples)
    y = np.linspace(bounds[2], bounds[3], num=y_samples or samples)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(shape=X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])
    return X, Y, Z


def contourf(fig, f, bounds=[-1, 1, -1, 1], limits=[-1, -0.5, 0, 0.5, 1],
             colors=('#FF6666', '#FF8888', '#8888FF', '#6666FF'),
             x_samples=50, y_samples=50):
    X, Y, Z = grid_function(f, bounds=bounds, x_samples=x_samples, y_samples=y_samples)
    
    return fig.contourf(X, Y, Z, limits, colors=colors, origin='lower', extend='both')


def contour(f, fig=plt, bounds=[-1, 1, -1, 1], limits=[0, 2],
            colors=None, samples=10,
            x_samples=50, y_samples=50):
    X, Y, Z = grid_function(f, bounds=bounds, samples=samples,
                            x_samples=x_samples, y_samples=y_samples)
    
    return fig.contour(X, Y, Z, np.linspace(limits[0], limits[1], num=samples), colors=colors, origin='lower', extend='both')


def heatmap(f, fig=plt, bounds=[-1, 1, -1, 1], limits=[-1, -0.5, 0, 0.5, 1],
             cmap='coolwarm', samples=50, x_samples=None, y_samples=None):
    X, Y, Z = predict_area(f, bounds=bounds, samples=samples,
                           x_samples=x_samples, y_samples=y_samples)
    Z[Z >  1] =  1
    Z[Z < -1] = -1
    return fig.imshow(Z, interpolation='bilinear', origin='lower', cmap=plt.get_cmap(cmap), extent=bounds)


def pcolor(fig, f, bounds=[-1, 1, -1, 1], cmap=cm.coolwarm, samples=50,
           x_samples=None, y_samples=None):
    X, Y, Z = grid_function(f, bounds=bounds, samples=samples,
                            x_samples=x_samples, y_samples=y_samples)
    #return fig.pcolor(X, Y, Z, cmap=cmap, vmin=-1, vmax=+1)
    return fig.pcolormesh(X, Y, Z, cmap=cmap, vmin=-1, vmax=+1)


def dataset(dataset, colors=None, markers=None,
            linewidth=None, size=None, margin=0, return_all=False):
    """Draw a classification dataset object.

    Args:
        dataset (:attr:`happyml.datasets.DataSet`): Dataset object.
        colors (list or tuple):
        markers (number, list or tuple):
        linewidth (number, list or tuple):
        size (number, list or tuple):
        margin (number):
        return_all (boolean): Returns all painted matplotlib objects
            (one per class) although there is no sample of that
            class the dataset. Defaults to False.

    """
    # Default params.
    theme = get_theme()
    colors = colors or theme["colors"]
    if isinstance(markers, (int, float)):
        markers = [markers] * 2
    else:
        markers = markers or theme["markers"]
    if isinstance(linewidth, (int, float)):
        linewidth = [linewidth] * 2
    else:
        linewidth = linewidth or theme["linewidth"]
    if isinstance(size, (int, float)):
        size = [size] * 2
    else:
        size = size or theme["size"]

    classes = dataset.Y.flatten().astype(int)
    classes[classes == -1] = 0
    
    scatters = []
    for i, c in enumerate(range(10)):
        idx = classes == c
        # plt.scatter per class. This allows us using a different
        # marker per class.
        options = {
            "zorder": 10 + i,
            "c": colors[i],
            "s": size[i],
            "linewidth": linewidth[i],
            "marker": markers[i],
            "picker": True,
        }
        if idx.any():
            scatters += [plt.scatter(dataset.X[idx, 0], dataset.X[idx, 1],
                                     **options)]
        elif return_all:
            scatters += [plt.scatter([], [], **options)]
    plt.margins(x=margin, y=margin)
    # Return all the painted objects.
    return scatters


def binary_ones(X, Y, Z, **kwargs):
    """Print the Z matrix assuming that contains numbers between -1 and 1.

    Keyword Arguments:
        autoscale (boolean): Defaults to True.
        colors (list): 2 colors, first to the -1 class, second to the
            +1 class. Defaults to lighted color classes 0 and 1.
        contours (boolean): Plot contour lines. Defaults to True.

    """
    autoscale = kwargs.get('autoscale', True)
    colors = kwargs.get('colors', binary_ones_colors)

    ax = plt.gca()
    ax.set_autoscale_on(autoscale)

    ax.margins(x=0., y=0.)
    ax.contourf(X, Y, Z, [-1, 0, 1], colors=colors,
            origin='lower', extend='both')
    if kwargs.get('contours', True):
        ax.contour(X, Y, Z, [0,], linewidths=3, colors='#000000')


def binary_margins(X, Y, Z, **kwargs):
    """Print the Z matrix assuming that contains real numbers
    of two classes: positive and negative.

    Between -1 and 1 uses lighters colors that represents the margin.

    Keyword Arguments:
        autoscale (boolean): Adapt the plot view to the values of the
            matrices ``X`` and ``Y``. Defaults to True.
        colors (list): 4 colors, first to the confident -1 class,
            second to the margin of the -1 class, third the margin
            of the +1 class and fourth the confident +1 class.
            Defaults to lighted colors for classes 0 and 1.
        contours (boolean): Plot contour lines. Defaults to True.

    """
    autoscale = kwargs.get('autoscale', True)
    colors = kwargs.get('colors', binary_margin_colors)
    levels = [-1.1, -1, 0, 1, 1.1]

    ax = plt.gca()
    ax.set_autoscale_on(autoscale)

    plt.margins(x=0., y=0.)
    plt.contourf(X, Y, Z, levels, colors=colors,
            origin='lower', extend='both')
    if kwargs.get('contours', True):
        plt.contour(X, Y, Z, [-1, 1], linewidths=1.5, colors='#000000',
                    linestyles='dashed', headwidth=10)
        plt.contour(X, Y, Z, [0,], linewidths=3, colors='#000000')

def model_binary_ones(model, data=None, **kwargs):
    if dataset is not None: dataset(data)
    X, Y, Z = predict_area(model, **kwargs)
    binary_ones(X, Y, Z, **kwargs)

def model_binary_margins(model, **kwargs):
    X, Y, Z = predict_area(model, **kwargs)
    binary_margins(X, Y, Z, **kwargs)

def show():
    plt.show()
