Explanation methods included
============================

``text_explainability`` includes methods for model-agnostic *local explanation* and *global explanation*. Each of these methods can be fully customized to fit the explainees' needs.

.. list-table:: Explanation methods in `text_explainability`
   :header-rows: 1

   * - Type
     - Explanation method
     - Description
     - Paper/link
   * - *Local explanation*
     - ``LIME``
     - Calculate feature attribution with *Local Intepretable Model-Agnostic Explanations* (LIME).
     - [\ `Ribeiro2016 <https://paperswithcode.com/method/lime>`_\ ], `interpretable-ml/lime <https://christophm.github.io/interpretable-ml-book/lime.html>`_
   * - 
     - ``KernelSHAP``
     - Calculate feature attribution with *Shapley Additive Explanations* (SHAP).
     - [\ `Lundberg2017 <https://paperswithcode.com/paper/a-unified-approach-to-interpreting-model>`_\ ], `interpretable-ml/shap <https://christophm.github.io/interpretable-ml-book/shap.html>`_
   * - 
     - ``LocalTree``
     - Fit a local decision tree around a single decision.
     - [\ `Guidotti2018 <https://paperswithcode.com/paper/local-rule-based-explanations-of-black-box>`_\ ]
   * - 
     - ``LocalRules``
     - Fit a local sparse set of label-specific rules using ``SkopeRules``.
     - `github/skope-rules <https://github.com/scikit-learn-contrib/skope-rules>`_
   * - 
     - ``FoilTree``
     - Fit a local contrastive/counterfactual decision tree around a single decision.
     - [\ `Robeer2018 <https://github.com/MarcelRobeer/ContrastiveExplanation>`_\ ]
   * - *Global explanation*
     - ``TokenFrequency``
     - Show the top-\ *k* number of tokens for each ground-truth or predicted label.
     - 
   * - 
     - ``TokenInformation``
     - Show the top-\ *k* token mutual information for a dataset or model.
     - `wikipedia/mutual_information <https://en.wikipedia.org/wiki/Mutual_information>`_
   * - 
     - ``KMedoids``
     - Embed instances and find top-\ *n* prototypes (can also be performed for each label using ``LabelwiseKMedoids``\ ).
     - `interpretable-ml/prototypes <https://christophm.github.io/interpretable-ml-book/proto.html>`_
   * - 
     - ``MMDCritic``
     - Embed instances and find top-\ *n* prototypes and top-\ *n* criticisms (can also be performed for each label using ``LabelwiseMMDCritic``\ ).
     - [\ `Kim2016 <https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html>`_\ ], `interpretable-ml/prototypes <https://christophm.github.io/interpretable-ml-book/proto.html>`_


Credits
-------


* Florian Gardin, Ronan Gautier, Nicolas Goix, Bibi Ndiaye and Jean-Matthieu Schertzer. *\ `Skope-rules <https://github.com/scikit-learn-contrib/skope-rules>`_\ *. 2020.
* Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Dino Pedreschi, Franco Turini and Fosca Gianotti. *\ `Local Rule-Based Explanations of Black Box Decision Systems <https://paperswithcode.com/paper/local-rule-based-explanations-of-black-box>`_\ *. 2018.
* Been Kim, Rajiv Khanna and Oluwasanmi O. Koyejo. `Examples are not Enough, Learn to Criticize! Criticism for Interpretability <https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html>`_. *Advances in Neural Information Processing Systems (NIPS 2016)*. 2016.
* Scott Lundberg and Su-In Lee. `A Unified Approach to Interpreting Model Predictions <https://paperswithcode.com/paper/a-unified-approach-to-interpreting-model>`_. *31st Conference on Neural Information Processing Systems (NIPS 2017)*. 2017.
* Christoph Molnar. *\ `Interpretable Machine Learning: A Guide for Making Black Box Models Explainable <https://christophm.github.io/interpretable-ml-book/>`_\ *. 2021.
* Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin. `"Why Should I Trust You?": Explaining the Predictions of Any Classifier <https://paperswithcode.com/method/lime>`_. *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2016)*. 2016.
* Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin. `Anchors: High-Precision Model-Agnostic Explanations <https://github.com/marcotcr/anchor>`_. *AAAI Conference on Artificial Intelligence (AAAI)*. 2018.
* Jasper van der Waa, Marcel Robeer, Jurriaan van Diggelen, Matthieu Brinkhuis and Mark Neerincx. `"Contrastive Explanations with Local Foil Trees" <https://github.com/MarcelRobeer/ContrastiveExplanation>`_. *2018 Workshop on Human Interpretability in Machine Learning (WHI 2018)*. 2018.
