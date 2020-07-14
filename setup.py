from setuptools import setup

setup(name='sui',
      version='0.1.8',
      description='Some Useful Implementations',
      long_description='SUI(Some Useful Implementations) is a peckage including selected implementations apropos machine learning, deep learning, and reinforcement learning algorithms.',
      url='http://github.com/l-tang/sui',
      author='Li Tang',
      author_email='litang1025@gmail.com',
      license='MIT',
      packages=['sui', 'sui.ml', 'sui.dl', 'sui.rl', 'sui.ds'],
      zip_safe=False)
