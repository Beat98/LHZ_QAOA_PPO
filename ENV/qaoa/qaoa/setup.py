from setuptools import setup

setup(name='qaoa',
      version='0.2.86',
      description='class to do qaoa stuff',
      url='https://gitlab.com/lecher-group/qaoa',
      author='Kilian Ender',
      author_email='kilian.ender@uibk.ac.at',
      packages=['qaoa', 'qaoa_analytics'],
      install_requires=[
          'qutip>=4.4.1', 'numpy', 'scipy', 'qiskit>=0.34'
      ],
      zip_safe=False)
