from setuptools import setup, find_packages


VERSION = "0.0.0"


def read_requirements():
    with open("requirements.txt") as file:
        return list(file)


def get_long_description():
    with open("README.md", encoding="utf8") as file:
        return file.read()


setup(
    name="lmt",
    description="The main CLI tool within the LLM-Toolbox, designed to enable seamless communication with ChatGPT. Customize your experience with tailored commands for various tasks, or use `lmt` standalone for direct interaction with ChatGPT.",
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
        ]
    },
    python_requires=">=3.8",
)

