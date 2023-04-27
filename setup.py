from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "AIMY evaluation and target shooting scripts"
LONG_DESCRIPTION = "AIMY evaluation and target shooting scripts"

setup(
    name="aimy_target_shooting",
    version=VERSION,
    author="Alexander Dittrich",
    author_email="alexander.dittrich@tuebingen.mpg.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license="BSD-3-Clause",
    keywords="AIMY evaluation and target shooting scripts",
    url="https://webdav.tuebingen.mpg.de/aimy/",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
