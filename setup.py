from setuptools import setup

setup(name='awkde',
      version='0.1',
      description='Adaptive width gaussian KDE',
      author='Thorben Menne',
      author_email='thorben.menne@tu-dortmund.de',
      url='github.com/mennthor/awkde',
      packages=['awkde'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      )
