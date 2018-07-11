import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="caver",
    version="0.0.1",
    author="Keming Yang",
    author_email="kemingy94@gmail.com",
    description="A toolkit for multiclass text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guokr/Caver",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ),
    entry_points={
        'console_scripts': [
            # 'trickster_train=trickster::train',
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'plane',
        'scikit-learn',
        'jieba',
        'torch',
    ]
)
