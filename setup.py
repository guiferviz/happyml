
"""HappyML is a Machine Learning library for educational purposes.

Author: guiferviz
GitHub: https://github.com/guiferviz/happyml-py
"""


from os import listdir
from os.path import join

from setuptools import setup


# Creates a __version__ variable.
exec(open("happyml/_version.py").read())


# Install all files under scripts dir.
SCRIPTS_DIR = "scripts"
SCRIPTS = [join(SCRIPTS_DIR, i) for i in listdir(SCRIPTS_DIR)]


setup(name='happyml',
      version=__version__,
      description='Machine Learning library for educational purposes.',
      long_description=open('README.rst').read(),
      keywords='happy machine learning',
      url='https://github.com/guiferviz/happyml-py',
      author='guiferviz',
      author_email='guiferviz@gmail.com',
      license='MIT',
      packages=['happyml'],
      install_requires=['numpy', 'matplotlib'],
      tests_require=['nose'],
      test_suite='nose.collector',
      zip_safe=False,
      scripts=SCRIPTS)
