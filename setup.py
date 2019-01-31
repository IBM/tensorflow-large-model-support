from setuptools import find_packages
from setuptools import setup

install_requires = [
    'tensorflow-gpu >= 1.5',
    'toposort >= 1.5',
]

setup(
    name='tensorflow-large-model-support',
    version='0.1.0',
    description='TensorFlow Large Model Support: graph editing library for large model support',
    author='Tung D. Le',
    author_email='tung@jp.ibm.com',
    packages=find_packages(),
    install_requires=install_requires
)
