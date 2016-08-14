
from os import listdir
from os.path import join

from setuptools import setup

from happyml import __version__


# Install all files under scripts dir.
SCRIPTS_DIR = "scripts"
scripts = [join(SCRIPTS_DIR, i) for i in listdir(SCRIPTS_DIR)]


setup(name='happyml',
      version=__version__,
      description='Machine Learning library for educational purpose.',
      keywords='happy machine learning',
      url='https://github.com/guiferviz/happyml-py',
      author='guiferviz',
      author_email='guiferviz@gmail.com',
      license='MIT',
      packages=['happyml'],
      install_requires=['numpy', 'matplotlib'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      scripts=scripts)
