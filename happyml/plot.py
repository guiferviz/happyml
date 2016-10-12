
"""
Plotting module.
"""

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.colors import rgb2hex
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread

import happyml


MAX_CLASS_NUMBER = 10


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


def get_themes():
    """Return a list with all the themes names."""
    themes_names = []
    for k in happyml.config["themes"].keys():
        themes_names += [k]
    return themes_names


def set_theme(theme_name):
    """Safe theme set.

    Performs ``happyml.config["theme"] = theme_name`` after
    checking that ``theme_name`` is the name of an existing
    theme.

    Args:
        theme_name (string): Name of the new theme.

    Raise:
        ValueError if you provide and unknow theme name.

    See Also:
        :attr:`happyml.plot.get_themes`

    """
    list_themes = get_themes()
    if theme_name not in list_themes:
        raise ValueError("Unknown theme '%s'. Use one of the following %s"\
                         " or create your own theme in 'happyml.conf'."
                         % (theme_name, repr(list_themes)))

    happyml.config["theme"] = theme_name


def get_theme(prop=None):
    """Return the current theme dictionary.

    The output dict contains all the info of the theme
    (colors, size, markers, alpha...).

    If you do not want the full theme dict you can use
    the prop argument to get only one prop of the current
    theme.

    Args:
        prop (string): Return the ``prop`` value of the
            current theme.

    Return:
        dictionary or theme property

    Example:
        .. code-block:: python

            # One way of getting theme colors:
            theme = get_theme()
            colors = theme["colors"]

            # The same in one line:
            colors = get_theme("colors")

    """
    name_theme = happyml.config["theme"]

    if prop:
        return happyml.config["themes"][name_theme][prop]

    return happyml.config["themes"][name_theme]


def get_class_color(n_class, format="hex"):
    """Return the color of the spedified class.

    This function looks up the theme colors table and
    returns the color of the spedified class.

    Args:
        n_class (number): Number from -1 to 9.
            -1 returns the same color as the 0 class.
        format (string): 'hex' or 'rgb'.

    Return:
        a string if format = 'hex' or rgb-tuple if
        format = 'rgb'.

    Raise:
        ValueError if you provide and unknow format.

    """
    if n_class == -1: n_class = 0

    if not (0 <= n_class < MAX_CLASS_NUMBER):
        raise ValueError("Class number out of range. n_class must be "\
                         "between -1 and %d" % (MAX_CLASS_NUMBER - 1))

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


def get_binary_ones_area_colors(classes):
    return (light_hex_color(get_class_color(classes[0]), light=0.3),
            light_hex_color(get_class_color(classes[1]), light=0.3))


def get_binary_margin_area_colors():
    return (light_hex_color(get_class_color(0), light=0.3),
            light_hex_color(get_class_color(0), light=0.4),
            light_hex_color(get_class_color(1), light=0.4),
            light_hex_color(get_class_color(1), light=0.3))


def predict_1d_area(model, limits=None, samples=50, **kwargs):
    if not limits:
        limits = [-1, 1]
    elif len(limits) != 2:
        raise ValueError("limits need 2 values: [xmin, xmax]")

    x = np.linspace(limits[0], limits[1], num=samples)
    x = x.reshape((samples, 1))

    y = model.predict(x)

    return x.flatten(), y


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


def prepare_plot(limits=None, scaled=True, autoscale=True,
                 margin=0, margin_x=None, margin_y=None,
                 grid=False, grid_x=None, grid_y=None,
                 ticks=True, off=False, xlabel=None, ylabel=None,
                 label_size=None, title=None, title_size=None, **kwargs):
    """Set basic properties of the matplotlib plot."""
    ax = plt.gca()

    if scaled: ax.axis('scaled')

    if autoscale is not None: ax.set_autoscale_on(autoscale)

    if margin is not None:
        margin_x = margin_x or margin
        margin_y = margin_y or margin
        ax.margins(x=margin_x, y=margin_y)

    if limits:
        if len(limits) != 4:
            raise ValueError("limits need 4 values: "\
                             "[xmin, xmax, ymin, ymax]")
        ax.set_xlim(limits[0:2])
        ax.set_ylim(limits[2:4])

    grid_x = grid_x or grid
    grid_y = grid_y or grid
    if grid or grid_y:
        ax.yaxis.grid()
    if grid or grid_x:
        ax.xaxis.grid()

    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if off:
        ax.axis("off")

    if xlabel is not None:
        if label_size is not None:
            ax.set_xlabel(xlabel, fontsize=label_size)
        else:
            ax.set_xlabel(xlabel)

    if ylabel is not None:
        if label_size is not None:
            ax.set_ylabel(ylabel, fontsize=label_size)
        else:
            ax.set_ylabel(ylabel)

    if title is not None:
        if title_size is not None:
            ax.set_title(title, fontsize=label_size)
        else:
            ax.set_title(title)


def dataset_continuous(dataset, **kwargs):
    """Draw a continuous dataset object.

    Args:
        dataset (:attr:`happyml.datasets.DataSet`): Dataset to draw.

    """
    if "continuous" not in dataset.get_type() or dataset.get_d() != 1:
        raise ValueError("Invalid dataset. Expecting a continuous"
                         " dataset with 1 input dimension")

    for i in range(dataset.get_k()):
        plt.scatter(dataset.X[:, i], dataset.Y[:, i])
    prepare_plot(**kwargs)


def dataset_classification(dataset, colors=None, markers=None,
                           alpha=None, linewidth=None, size=None,
                           return_all=False, **kwargs):
    """Draw a classification dataset object.

    Args:
        dataset (:attr:`happyml.datasets.DataSet`): Dataset to draw.
        colors (list or tuple):
        markers (number, list or tuple):
        alpha (number, list or tuple): Any value between 1 (totally
            visible) and 0 (invisible).
        linewidth (number, list or tuple):
        size (number, list or tuple):
        return_all (boolean): Returns all painted matplotlib objects
            (one per class) although there is no sample of that
            class the dataset. Defaults to False.

    """
    # Default params.
    theme = get_theme()
    if isinstance(colors, str):
        colors = [colors]
    colors = colors or theme["colors"]
    if isinstance(markers, (int, float)):
        markers = [markers]
    markers = markers or theme["markers"]
    if isinstance(linewidth, (int, float)):
        linewidth = [linewidth]
    linewidth = linewidth or theme["linewidth"]
    if isinstance(size, (int, float)):
        size = [size]
    size = size or theme["size"]
    if isinstance(alpha, (int, float)):
        alpha = [alpha]
    alpha = alpha or theme["alpha"]

    classes = dataset.Y.flatten().astype(int)
    classes[classes < 0] = 0
    unique_classes = np.unique(classes)
    if return_all:
        # FIXME: limited class to plot.
        # Transform return_all to return_at_least
        # that contains the number of classes the user
        # want to receive.
        unique_classes = range(MAX_CLASS_NUMBER)
    
    scatters = []
    for c in unique_classes:
        idx = classes == c
        # plt.scatter per class. This allows us using a different
        # options per class.
        options = {
            "zorder": 10 + c,
            "c": colors[c % len(colors)],
            "s": size[c % len(size)],
            "linewidth": linewidth[c % len(linewidth)],
            "marker": markers[c % len(markers)],
            "alpha": alpha[c % len(alpha)],
            "picker": True,
        }
        if np.any(idx):
            scatters += [plt.scatter(dataset.X[idx, 0], dataset.X[idx, 1],
                                     **options)]
        elif return_all:
            scatters += [plt.scatter([], [], **options)]

    prepare_plot(**kwargs)
    # Return all the painted objects.
    return scatters


def dataset(dataset, dtype=None, **kwargs):
    dtype = dtype or dataset.get_type()
    if "continuous" in dtype:
        return dataset_continuous(dataset, **kwargs)
    elif "binary" in dtype or "multiclass" in dtype:
        return dataset_classification(dataset, **kwargs)
    else:
        raise ValueError("Dataset of type '%s' cannot be plotted" %
                         dtype)


def binary_ones(X, Y, Z, fill=True, colors=None, class_colors=None,
                contour=True, contour_width=3, contour_color=None,
                **kwargs):
    """Paint the Z matrix using two colors, one for the positive
    numbers and the other one for the negative.

    The matrices X and Y define a rectangular area and the Z
    matrix represents some output inside that area.

    You can use any param used in :attr:`happyml.plot.prepare_plot`.

    Args:
        X (numpy.ndarray): 'x' coordinate.
        Y (numpy.ndarray): 'y' coordinate.
        Z (numpy.ndarray): Matrix to draw.
        fill (boolean): Fill with colors the positive and negative
            areas. Defaults to True.
        colors (list): 2 colors, first to the -1 class, second to the
            +1 class. Defaults to lighted color classes 0 and 1.
        class_colors (list): 2 class numbers. Lighted colors of this
            2 classes will be used. Defaults to [0, 1].
        contour (boolean): Plot a contour line between positive
            and negative numbers. Defaults to True.
        contour_width (number): Width of the contour line.
            Defaults to 3.
        contour_color (string): Color of the contour line in
            hexadecimal. Defaults to black (#000000).

    See Also:
        :attr:`happyml.plot.prepare_plot`

    """
    class_colors = class_colors or [0, 1]
    colors = colors or get_binary_ones_area_colors(class_colors)
    contour_color = contour_color or "#000000"

    prepare_plot(**kwargs)

    if fill:
        plt.contourf(X, Y, Z, [-1, 0, 1], colors=colors,
            origin='lower', extend='both')
    if contour:
        plt.contour(X, Y, Z, [0,], linewidths=contour_width,
            colors=contour_color)


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
    colors = kwargs.get('colors', get_binary_margin_area_colors())
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


def plot_line(x, y, **kwargs):
    plt.plot(x, y)
    prepare_plot(**kwargs)


def model_binary_ones(model, **kwargs):
    X, Y, Z = predict_area(model, **kwargs)
    binary_ones(X, Y, Z, **kwargs)


def model_binary_margins(model, **kwargs):
    X, Y, Z = predict_area(model, **kwargs)
    binary_margins(X, Y, Z, **kwargs)


def model_line(model, **kwargs):
    x, y = predict_1d_area(model, **kwargs)
    plot_line(x, y, **kwargs)


def model(model, plot_type=None, data=None, **kwargs):
    if data is not None: dataset(data)
    plot_type = plot_type or model._plot_type
    if plot_type:
        if "binary_one" in plot_type:
            return model_binary_ones(model, **kwargs)
        elif "binary_margin" in plot_type:
            return model_binary_margins(model, **kwargs)
        elif "line" in plot_type:
            return model_line(model, **kwargs)
    raise ValueError("Model of type '%s' cannot be plotted" %
                     plot_type)


def imshow(img, **kwargs):
    """Show an image in a figure.

    Args:
        img (str or numpy.ndarray):

    Example:
        .. code-block:: python

            img = imread("myimg.png")
            imshow(img)

            # The same in one line:
            imshow("myimg.png")

    """
    if type(img) == str:
        img = imread(img)

    kwargs.setdefault("off", True)
    prepare_plot(**kwargs)
    plt.imshow(img)


def figure(*args, **kwargs):
    return plt.figure(*args, **kwargs)


def subplot(*args, **kwargs):
    return plt.subplot(*args, **kwargs)


def suptitle(*args, **kwargs):
    return plt.suptitle(*args, **kwargs)


from models import Model
Model.plot = model
Model._plot_type = None

from models import Perceptron, PerceptronKernel, LinearRegression
Perceptron._plot_type = "binary_ones"
PerceptronKernel._plot_type = "binary_ones"
LinearRegression._plot_type = "line"

from datasets import DataSet
DataSet.plot = dataset


def show():
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
