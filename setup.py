from setuptools import setup, find_packages

def load_requirements(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f]

setup(
    name='WindML',
    version='0.1',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt')
)

