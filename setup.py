from setuptools import find_packages, setup

VERSION = "0.0.27"


def read_requirements():
    with open("requirements.txt") as file:
        return list(file)


def get_long_description():
    with open("README.md", encoding="utf8") as file:
        return file.read()


setup(
    name="lmt-cli",
    description="""
The primary CLI of LLM-Toolbox facilitates direct interaction with ChatGPT from your terminal, offering customizable templates for message generation.
""",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Sébastien De Revière",
    url="https://github.com/sderev/lmt",
    project_urls={
        "Documentation": "https://github.com/sderev/lmt",
        "Issues": "http://github.com/sderev/lmt/issues",
        "Changelog": "https://github.com/sderev/lmt/releases",
    },
    license="Apache Licence, Version 2.0",
    version=VERSION,
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "lmt=lmt_cli.cli:lmt",
        ]
    },
    python_requires=">=3.8",
)
