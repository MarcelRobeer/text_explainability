import setuptools

setuptools.setup( # type: ignore
    name = 'explainability',
    version = '0.1',
    description = '',
    long_description = '',
    author = 'Marcel Robeer',
    author_email = 'm.j.robeer@uu.nl',
    classifiers = [
        'Programming Language :: Python'
    ],
    packages = setuptools.find_packages(), # type : ignore
    python_requires = '>=3.8'
)