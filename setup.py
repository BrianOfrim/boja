import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boja",
    version="0.0.1",
    author="Brian Ofrim",
    author_email="bofrim@ualberta.ca",
    description="An end to end object detection tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrianOfrim/boja",
    packages=setuptools.find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires=">=3.6",
)
