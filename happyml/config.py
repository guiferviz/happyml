

import ast
import os
import warnings

from pkg_resources import Requirement, resource_filename
from ConfigParser import ConfigParser


CONFIG_FILENAME = "happyml.conf"

ERROR_CONFIG_FILE = "Error parsing config file %s."

DEFAULT_CONFIG = {
    "recursive": False,

    "logging": {
        "level": "debug"
    },

    "themes": {
        "default": {
            "colors": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                       "#ffff33", "#a65628", "#f781bf", "#999999", "#000000"],
            "markers": ["x", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            "linewidth": [0.25, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25, 0.25],
            "size": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        },
        "set2": {
            "colors": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
                       "#ffd92f", "#e5c494", "#b3b3b3", "#999999", "#000000"],
            "markers": ["x", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
            "linewidth": [0.25, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25, 0.25],
            "size": [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
        }
    },

    "theme": "default",
}


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

    Return:
        dictionary: Dict with all pairs key-value
        in file. Every value is a string.

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