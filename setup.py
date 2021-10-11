from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='matrixstitcher',
   version='0.2.2',
   description='An Elementary-transform Tape System for Linear Algebra Education Purpose',
   license="MIT",
   long_description=long_description,
   author='Raion',
   author_email='junhaoliu17@gmail.com',
   packages=['matrixstitcher'],
   install_requires=['numpy']
)
