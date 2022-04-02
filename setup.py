from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name="nfbq",
    version="0.0.1",
    description="Normalizing flow variational inference for expensive posteriors",
    packages=find_packages(where='nfbq'),
    author="Danilo de Freitas Naiff",
    author_email="dfnaiff@gmail.com",
    url="https://github.com/DFNaiff/NFBQ/"
)
