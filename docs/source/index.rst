.. image:: https://git.science.uu.nl/m.j.robeer/text_explainability/-/raw/main/img/TextLogo-Logo%20large.png
    :alt: T_xt Explainability logo
    :align: center


A generic explainability architecture for explaining text machine learning models
---------------------------------------------------------------------------------


.. image:: https://img.shields.io/pypi/v/text_explainability
   :target: https://pypi.org/project/text-explainability/
   :alt: PyPI


.. image:: https://pepy.tech/badge/text-explainability
   :target: https://pepy.tech/project/text-explainability
   :alt: Downloads


.. image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
   :target: https://pypi.org/project/text-explainability/
   :alt: Python_version


.. image:: https://img.shields.io/badge/build-passing-brightgreen
   :target: https://git.science.uu.nl/m.j.robeer/text_explainability/-/pipelines
   :alt: Build_passing


.. image:: https://img.shields.io/pypi/l/text_explainability
   :target: https://www.gnu.org/licenses/lgpl-3.0.en.html
   :alt: License


.. image:: https://img.shields.io/badge/docs-external-blueviolet
   :target: https://marcelrobeer.github.io/text_explainability
   :alt: Docs_passing


.. image:: https://img.shields.io/badge/code%20style-flake8-aa0000
   :target: https://github.com/PyCQA/flake8
   :alt: Code style: black


----

``text_explainability`` provides a **generic architecture** from which well-known state-of-the-art explainability approaches for text can be composed. This modular architecture allows components to be swapped out and combined, to **quickly develop new types of explainability approaches** for (natural language) text, or to **improve a plethora of approaches by improving a single module**.

Several example methods are included, which provide **local explanations** (\ *explaining the prediction of a single instance*\ , e.g. ``LIME`` and ``SHAP``\ ) or **global explanations** (\ *explaining the dataset, or model behavior on the dataset*\ , e.g. ``TokenFrequency`` and ``MMDCritic``\ ). By replacing the default modules (e.g. local data generation, global data sampling or improved embedding methods), these methods can be improved upon or new methods can be introduced.

|copy| Marcel Robeer, 2021-2022

Quick tour
----------

**Local explanation**\ : explain a models' prediction on a given sample, self-provided or from a dataset.

.. code-block:: python

   from text_explainability import LIME, LocalTree

   # Define sample to explain
   sample = 'Explain why this is positive and not negative!'

   # LIME explanation (local feature importance)
   LIME().explain(sample, model).scores

   # List of local rules, extracted from tree
   LocalTree().explain(sample, model).rules

**Global explanation**\ : explain the whole dataset (e.g. train set, test set), and what they look like for the ground-truth or predicted labels.

.. code-block:: python

   from text_explainability import import_data, TokenFrequency, MMDCritic

   # Import dataset
   env = import_data('./datasets/test.csv', data_cols=['fulltext'], label_cols=['label'])

   # Top-k most frequent tokens per label
   TokenFrequency(env.dataset).explain(labelprovider=env.labels, explain_model=False, k=3)

   # 2 prototypes and 1 criticisms for the dataset
   MMDCritic(env.dataset)(n_prototypes=2, n_criticisms=1)


Using text_explainability
-------------------------
:doc:`installation`
    Installation guide, directly installing it via `pip`_ or through the `git`_.

:doc:`example-usage`
    An extended usage example.

:doc:`explanation-methods`
    Overview of the explanation methods included in ``text_explainability``.

:doc:`text_explainability API reference <api/text_explainability>`
    A reference to all classes and functions included in the ``text_explainability``.


Development
-----------
`text_explainability @ GIT`_
    The `git`_ includes the open-source code and the most recent development version.

:doc:`changelog`
    Changes for each version are recorded in the changelog.

:doc:`contributing`
    Contributors to the open-source project and contribution guidelines.


Extensions
----------

.. image:: https://git.science.uu.nl/m.j.robeer/text_sensitivity/-/raw/main/img/TextLogo-Logo_large_sensitivity.png
    :alt: text_sensitivity logo
    :target: https://marcelrobeer.github.io/text_sensitivity/
    
``text_explainability`` can be extended to also perform *sensitivity testing*\ , checking for machine learning model robustness and fairness. The ``text_sensitivity`` package is available through `PyPI <https://pypi.org/project/text-sensitivity/>`_ and fully documented at `https://marcelrobeer.github.io/text_sensitivity/ <https://marcelrobeer.github.io/text_sensitivity/>`_.

Citation
--------

.. code-block:: bibtex

   @misc{text_explainability,
     title = {Python package text\_explainability},
     author = {Marcel Robeer},
     howpublished = {\url{https://git.science.uu.nl/m.j.robeer/text_explainability}},
     year = {2021}
   }


.. |copy|   unicode:: U+000A9 .. COPYRIGHT SIGN
.. _pip: https://pypi.org/project/text_explainability/
.. _git: https://github.com/MarcelRobeer/text_explainability
.. _`text_explainability @ GIT`: https://github.com/MarcelRobeer/text_explainability

.. toctree::
   :maxdepth: 1
   :caption: Using text_explainability
   :hidden:

   installation.rst
   example-usage.rst
   explanation-methods.rst

.. toctree::
   :maxdepth: 4
   :caption: API reference
   :hidden:

   api/text_explainability.rst

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   changelog.rst
   contributing.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
