import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="envtext",
    version="0.0.10",
    author="Bi Huaibin",
    author_email="bi.huaibin@foxmail.com",
    description="envtext for Chinese texts analysis in Environment domain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/celtics1863/envtext",
    project_urls={
        "Bug Tracker": "https://github.com/celtics1863/envtext/issues",
    },
    install_requires=[
        'jieba',
        'datasets',
        'gensim',
        'tqdm',
        'numpy',
        'pytorch-crf',
        'pandas',
        'torch',
        'transformers',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='NLP,bert,Chinese,LSTM,RNN,domain text analysis',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "": ["src/envtext/files/*"],
    },
    python_requires=">=3.6",
)