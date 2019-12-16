from setuptools import setup, find_packages

setup(name='subsets',
      version='0.0.1',
      description='Reparameterizable Subset Sampling',
      url='https://github.com/ermongroup/subsets',
      author='Sang Michael Xie',
      author_email='xie@cs.stanford.edu',
      packages=find_packages(),
      install_requires=[
        'tensorflow-gpu==1.15.0',
        'matplotlib',
        'numpy==1.14.5',
        'pandas',
        'scikit-learn',
        'scipy',
        'click',
        'keras',
        'nltk',
        'beautifulsoup4',
        'torch',
        'torchvision',
      ]
)
