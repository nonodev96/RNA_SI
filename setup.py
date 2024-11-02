"""Python setup.py for rna_si package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("rna_si", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="rna_si",
    version=read("rna_si", "VERSION"),
    description="Awesome rna_si created by nonodev96",
    url="https://github.com/nonodev96/RNA_SI/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="nonodev96",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["rna_si = rna_si.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
