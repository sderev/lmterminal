from setuptools import find_packages, setup

VERSION = "0.0.35"


def read_requirements():
    with open("requirements.txt") as file:
        return list(file)


def get_long_description():
    with open("README.md", encoding="utf8") as file:
        return file.read()


setup(
    name="LMterminal",
    description="Interact with OpenAI's ChatGPT models from the comfort of your terminal.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Sébastien De Revière",
    url="https://github.com/sderev/lmterminal",
    project_urls={
        "Documentation": "https://github.com/sderev/lmterminal",
        "Issues": "http://github.com/sderev/lmterminal/issues",
        "Changelog": "https://github.com/sderev/lmterminal/releases",
    },
    license="Apache Licence, Version 2.0",
    version=VERSION,
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "lmt=lmterminal.cli:lmt",
        ]
    },
    python_requires=">=3.8",
)
