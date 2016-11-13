

import ast
import os
import warnings

from pkg_resources import Requirement, resource_filename
from ConfigParser import ConfigParser


DATA_DIR = ".happyml"

CONFIG_FILENAME = "happyml.conf"

ERROR_CONFIG_FILE = "Error parsing config file %s."

DEFAULT_CONFIG = {
    "recursive": False,

    "logging": {
        "level": "debug"
    },

    "themes": {
        "default": {
            "colors": ["#ff5a5a", "#5c5cff", "#4daf4a", "#984ea3", "#ff7f00",
                       "#ffff33", "#a65628", "#f781bf", "#999999", "#000000"],
            "markers": ["x", "o", "*", "+", "h", "s", "8", "p", "D", "o"],
            "linewidth": [3, 0.25, 0.25, 3, 0.25,
                          0.25, 0.25, 0.25, 0.25, 0.25],
            "size": [60, 50, 100, 100, 50, 50, 50, 50, 50, 50],
            "alpha": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        "minus_plus": {
            "colors": ["#ff5a5a", "#4daf4a"],
            "markers": ["_", "+"],
            "linewidth": [3, 3],
            "size": [100, 100],
            "alpha": [1, 1],
        },
        "set2": {
            "colors": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
                       "#ffd92f", "#e5c494", "#b3b3b3", "#999999", "#000000"],
            "markers": ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            "linewidth": [0.25, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25, 0.25],
            "size": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        }
    },

    "theme": "default",
}


def happyml_data_dir():
    """Return the path of the data directory.

    The data directory is used to store downloaded datasets like MNIST.

    """
    return os.path.join(os.path.expanduser("~"), DATA_DIR)


def happyml_config_files():
    """Get a lists with all the possible locations of
    the configs files.

    This method does not check the existence of the config files,
    it only returns a list with the paths.

    The files locations are returned in the following order:

    - ``./happyml.conf``
    - ``~/happyml.conf``
    - ``$HAPPYML_CONF/happyml.conf`` if the env var HAPPYML_CONF is defined
    - ``<PACKAGE_DATA>/happyml.conf``

    See Also:
        :attr:`read_config`

    """
    # Home and current dir config files.
    paths = [os.path.join(os.curdir, CONFIG_FILENAME),
             os.path.join(os.path.expanduser("~"), CONFIG_FILENAME)]
    # Environ HAPPYML_CONF config file.
    if os.environ.get("HAPPYML_CONF"):
        paths += [os.path.join(os.environ.get("HAPPYML_CONF"),
                               CONFIG_FILENAME)]
    # Default config file.
    paths += [resource_filename(Requirement.parse("happyml"),
                                CONFIG_FILENAME)]

    return paths


def read_config_file(filename):
    """Read a config file.

    If the config file has an invalid syntax or does
    not exists ``None`` is returned.

    If there is an error in the configuration file
    a warning is shown.

    Return:
        dictionary or ``None``.

    """
    dic = None

    if os.path.isfile(filename):
        file_txt = open(filename).read()
        try:
            dic = ast.literal_eval(file_txt)
        except:
            warnings.warn(ERROR_CONFIG_FILE % filename)

    return dic


def read_config():
    """Return a dict with all the configuration options.

    Return:
        :

        **config_dict** (dictionary): Dictionary with all the\
            config options.

        **read_files** (list): List with all the config files\
            which have been used to complete the config\
            dictionary.

    See Also:
        :attr:`happyml_config_files`

    """
    dic = DEFAULT_CONFIG.copy()

    paths = happyml_config_files()
    read_files = []
    for file in paths:
        dic_file = read_config_file(file)

        if dic_file:
            dic.update(dic_file)
            read_files += file
            if not dic.get("recursive"):
                break

    return dic, read_files
