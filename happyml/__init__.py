
"""Machine Learning library for academic purposes.

GitHub: https://github.com/guiferviz/happyml-py
"""


from pkg_resources import Requirement, resource_filename
from ConfigParser import ConfigParser
import os


__version__ = "0.0.5"
__author__  = "guiferviz"


CONFIG_FILENAME = "happyml.conf"


def happyml_config_files():
    """Get a lists with all the possible locations of
    the configs files.

    This method does not check the existence of the config files,
    it only returns a lists with the paths.

    The files locations are returned in the following order:

    - ``<PACKAGE_DATA>/happyml.conf``
    - ``$HAPPYML_CONF/happyml.conf`` if the env var HAPPYML_CONF is defined
    - ``~/happyml.conf``
    - ``./happyml.conf``

    See Also:
        :attr:`read_config`

    """
    # Default config file.
    paths = [resource_filename(Requirement.parse("happyml"),
                               CONFIG_FILENAME)]
    # Environ HAPPYML_CONF config file.
    if os.environ.get("HAPPYML_CONF"):
        paths += [os.path.join(os.environ.get("HAPPYML_CONF"),
                               CONFIG_FILENAME)]
    # Home and current dir config files.
    paths += [os.path.join(os.path.expanduser("~"), CONFIG_FILENAME),
              os.path.join(os.curdir, CONFIG_FILENAME)]

    return paths


def read_config():
    """Return a dict with all the configuration options.

    Return:
        config_dict (dictionary): Dictionary with all the\
            config options.

        read_files (list): List with all the config files\
            which have been used to complete the config\
            dictionary.

    See Also:
        :attr:`happyml_config_files`

    """
    config = ConfigParser()
    paths = happyml_config_files()
    read_files = config.read(paths)
    config_dict = dict(config.items("global"))

    return config_dict, read_files


def greet():
    """Prints a happy message to the standard output.

    HappyML is so friendly that greets you!
    """
    print "Those about to learn we salute you :)"


options, config_files_used = read_config()
