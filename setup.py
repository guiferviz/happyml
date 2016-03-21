
from setuptools import setup


setup(name='happyml',
      version='0.0.1',
      description='Machine Learning library for educational purpose.',
      keywords='happy machine learning',
      url='https://github.com/guiferviz/happyml-py',
      author='guiferviz',
      author_email='programmingh@gmail.com',
      license='MIT',
      packages=['happyml'],
      install_requires=['numpy'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
