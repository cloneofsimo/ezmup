import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="ezmup",
    py_modules=["ezmup"],
    version="0.0.1",
    description="Really Easy MuP",
    author="Simo Ryu",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt")),
        )
    ],
    include_package_data=True,
)
