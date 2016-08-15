
from os import listdir
from os.path import join

from setuptools import setup

from happyml import __version__


# Install all files under scripts dir.
SCRIPTS_DIR = "scripts"
scripts = [join(SCRIPTS_DIR, i) for i in listdir(SCRIPTS_DIR)]

# Read requirements.
install_requires = [line for line in
            open("requirements.txt").read().splitlines() if line]
tests_require = [line for line in
            open("requirements-tests.txt").read().splitlines() if line]

setup(name='happyml',
      version=__version__,
      description='Machine Learning library for educational purpose.',
      long_description=open('README.rst').read(),
      keywords='happy machine learning',
      url='https://github.com/guiferviz/happyml-py',
      author='guiferviz',
      author_email='guiferviz@gmail.com',
      license='MIT',
      packages=['happyml'],
      install_requires=install_requires,
      tests_require=tests_require,
      test_suite='nose.collector',
      zip_safe=False,
      scripts=scripts)
