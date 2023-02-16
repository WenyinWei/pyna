import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyna-chaos",
    version="0.0.2",
    author="Wenyin Wei",
    author_email="wenyin.wei.ww@gmail.com",
    description="Python package for DYNAmics (dynamical systems)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenyinWei/pyna",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "scipy", "sympy", "numpy", "pandas", # fundamental python packages
        "matplotlib", "plotly",
        "jupyter", "notebook", "ipympl", # "ipympl" for %matplotlib widget
        # "sparse", # for storing data of high order derivatives inter-relation
    ]
)