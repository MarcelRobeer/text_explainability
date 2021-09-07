import setuptools
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup( # type: ignore
    name = 'text_explainability',
    version = '0.3.7',
    description = 'Generic explainability architecture for text machine learning models',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Marcel Robeer',
    author_email = 'm.j.robeer@uu.nl',
    license = 'GNU LGPL v3',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    url='https://git.science.uu.nl/m.j.robeer/text_explainability',
    packages = setuptools.find_packages(), # type : ignore
    install_requires = [
        'instancelib>=0.3.2.0',
        'numpy>=1.19.5',
        'python-i18n>=0.3.9',
        'scikit-learn>=0.24.1',
    ],
    python_requires = '>=3.8'
)