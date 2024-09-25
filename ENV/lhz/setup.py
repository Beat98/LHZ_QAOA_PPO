from setuptools import setup

setup(name='lhz',
      version='0.2.2',
      description='Tools for working with the LHZ-architecture',
      url='https://gitlab.com/lecher-group/lhz',
      author='Kilian Ender',
      author_email='kilian.ender@uibk.ac.at',
      packages=['lhz'],
      install_requires=[
          'qutip', 'numpy', 'networkx'
      ],
      zip_safe=False)
