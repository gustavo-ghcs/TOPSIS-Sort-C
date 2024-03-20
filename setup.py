from setuptools import setup, find_packages

setup(
    name='TOPSIS-Sort-C',
    version='0.1',
    description='TOPSIS-Sort-C implemented in python',
    author='Gustavo Hollanda, Geovanna Domingos, Giovanna Machado, Higor Cunha, Maria Eduarda Melo',
    url='https://github.com/gustavo-ghcs/TOPSIS-Sort-C',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
    ],
)