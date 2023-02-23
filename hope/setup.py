from setuptools import setup, find_packages

# TODO: add other dependencies as well

setup(
   name='hope',
   version='0.0.11',
   packages=find_packages(
      exclude=['examples', 'scripts'], 
   ),
   install_requires=[
            "torch",
            "cv2",
            "timm",
            "mmcv",
            "mmsegmentation"
   ],
)