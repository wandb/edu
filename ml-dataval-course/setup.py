from setuptools import setup, find_packages

setup(
    name="ml-dataval-tutorial",
    version="0.1",
    author="Shreya Shankar",
    author_email="shreyashankar@berkeley.edu",
    url="https://github.com/shreyashankar/ml-dataval-tutorial",
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    scripts=[
        "download.sh",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
