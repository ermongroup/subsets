from setuptools import setup, find_packages

setup(name='subsets',
      version='0.0.1',
      description='Reparameterizable Subset Sampling',
      url='https://github.com/ermongroup/subsets',
      author='Sang Michael Xie',
      author_email='xie@cs.stanford.edu',
      packages=find_packages(),
      install_requires=[
        'tensorflow-gpu==2.9.3',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn==0.21.2',
        'scipy==1.3.0',
        'click==7.0',
        'keras==2.2.4',
        'nltk==3.4.3',
        'beautifulsoup4==4.7.1',
        'torch==1.1.0',
        'torchvision==0.3.0',
      ]
)
