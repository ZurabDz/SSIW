from setuptools import setup, find_packages


setup(
   name='hope',
   version='0.0.11',
   packages=find_packages(
      exclude=['examples', 'scripts'], 
   ),
)