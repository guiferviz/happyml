
"""Machine Learning library for academic purposes.

GitHub: https://github.com/guiferviz/happyml-py
"""


from config import read_config
from _version import __version__


__author__  = "guiferviz"


def greet():
    """Prints a happy message to the standard output.

    HappyML is so friendly that greets you!
    """
    print "Those about to learn we salute you :)"


config, config_files_used = read_config()


#import happyml.models as models
from models import *
import datasets
import plot
