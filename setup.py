from distutils.util import convert_path
from os import path

import setuptools

main_ns = {}
with open(convert_path('text_explainability/__init__.py')) as ver_file:
    exec(ver_file.read(), main_ns)

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup( # type: ignore
    name = 'text_explainability',
    version = main_ns['__version__'],
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
    url = 'https://git.science.uu.nl/m.j.robeer/text_explainability',
    packages = setuptools.find_packages(), # type : ignore
    include_package_data = True,
    install_requires = [
        'instancelib>=0.3.2.1',
        'numpy>=1.19.5',
        'python-i18n>=0.3.9',
        'scikit-learn>=0.24.1',
        'sentence-transformers',  # optional in future
        'scikit-learn-extra',  # optional in future
        'skope-rules>=1.0.1',  # optional in future
    ],
    python_requires = '>=3.8'
)
