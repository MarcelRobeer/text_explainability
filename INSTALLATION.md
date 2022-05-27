# Installation
Installation of `text_explainability` requires `Python 3.8` or higher.

### 1. Python installation
Install Python on your operating system using the [Python Setup and Usage](https://docs.python.org/3/using/index.html) guide.

### 2. Installing `text_explainability`
`text_explainability` can be installed:

* _using_ `pip`: `pip3 install` (released on [PyPI](https://pypi.org/project/text-explainability/))
* _locally_: cloning the repository and using `python3 setup.py install`

#### Using `pip`
1. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows)
2. Run the command:
    - `pip3 install text_explainability`, or
    - `pip install text_explainability`.

```console
user@terminal:~$ pip3 install text_explainability
Collecting text_explainability
...
Installing collected packages: text-explainability
Successfully installed text-explainability
```

> Speeding up the explanation-generation process can be done by using `pip3 install text_explainability[fast]` or having `fastcountvectorizer` installed.

#### Locally
1. Download the folder from `GitLab/GitHub`:
    - Clone this repository, or 
    - Download it as a `.zip` file and extract it.
2. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows) and navigate to the folder you downloaded `text_explainability` in.
3. In the main folder (containing the `setup.py` file) run:
    - `python3 setup.py install`, or
    - `python setup.py install`.

```console
user@terminal:~$ cd ~/text_explainability
user@terminal:~/text_explanability$ python3 setup.py install
running install
running bdist_egg
running egg_info
...
Finished processing dependencies for text-explainability
```
