"""Extension of `genbase.ui.notebook` for custom rendering of `text_explainability."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from genbase.ui import format_instances, get_color
from genbase.ui.notebook import Render as BaseRender
from genbase.ui.notebook import format_label
from genbase.ui.plot import plotly_available
from genbase.utils import package_available

MAIN_COLOR = '#1976D2'
POSITIVE_COLOR = '#4DAE52'
POSITIVE_COLOR_L = '#81C784'
NEGATIVE_COLOR = '#D93232'
NEGATIVE_COLOR_L = '#E57373'
TRANSLATION_DICT = {'lime': ('LIME', 'https://christophm.github.io/interpretable-ml-book/lime.html'),
                    'shap': ('SHAP', 'https://christophm.github.io/interpretable-ml-book/shap.html'),
                    'kernel_shap': ('KernelSHAP', 'https://christophm.github.io/interpretable-ml-book/shap.html'),
                    'foil_tree': ('Foil Trees', 'https://arxiv.org/abs/1806.07470'),
                    'local_tree': ('Build your own LIME (tree surrogate)', 'https://arxiv.org/abs/1910.13016'),
                    'skoperulesclassifier': ('Local Skope Rules classifier', 'https://dropsofai.com/mining-interpretable-rules-from-classification-models/'),  # noqa: E501
                    'baylime': ('BayLIME', 'https://proceedings.mlr.press/v161/zhao21a.html'),
                    'mutual_information': ('mutual information', 'https://en.wikipedia.org/wiki/Mutual_information'),
                    'kmedoids': ('KMedoids', 'https://christophm.github.io/interpretable-ml-book/proto.html'),
                    'mmdcritic': ('MMDCritic', 'https://christophm.github.io/interpretable-ml-book/proto.html')}


def default_renderer(meta: dict, content: dict, **renderargs) -> str:
    """Default renderer fallback."""
    return f'<p>{content}</p>'


def plotly_fallback(function):
    """Return a graphics renderer, with fallback if plotly is not available."""
    def inner(*args, **kwargs):
        return function(*args, **kwargs) if not plotly_available() else default_renderer(*args, **kwargs)
    return function


def get_meta_descriptors(meta: dict) -> Tuple[str]:
    """Get type, subtype & method from `meta`.

    Args:
        meta (dict): [description]

    Returns:
        Tuple[str]: type, subtype, method
    """
    def fmt(x):
        return str(x).strip().lower().replace(' ', '_')

    def get_from_meta(key: str) -> str:
        return fmt(meta[key]) if key in meta else ''

    return get_from_meta('type'), get_from_meta('subtype'), get_from_meta('method')


def original_scores_renderer(original_scores: dict, **renderargs) -> str:
    """Render predicted output scores of model on an instance."""
    def format_kv(k, v):
        return f'<tr><td><kbd>{k}</kbd></td><td>{v:.3f}</td><td style="text-align: left"><div title="{v}", ' + \
            'style="display: inline-block; background-color: black; height: 1em; ' + \
            f'width: {min(max(3, v * 100), 100)}px"></div></td></tr>'

    html = [format_kv(k, v) for k, v in original_scores.items()]
    return '<p>The model predicted the following scores for the instance:</p><table>' + '\n'.join(html) + '</table>'


def feature_attribution_renderer(meta: dict, content, **renderargs) -> str:
    """Render feature attribution return types."""
    import uuid

    min_value = renderargs.pop('min_value', -1.0)
    max_value = renderargs.pop('max_value', 1.0)
    colorscale = renderargs.pop('colorscale', [(0.0, NEGATIVE_COLOR_L), (0.5, '#eee'), (1.0, POSITIVE_COLOR_L)])

    id = 'id' + str(uuid.uuid4())
    shap_bool, plotly_bool = plotly_available(), package_available('shap')

    if plotly_bool:
        import plotly.express as px
        from genbase.ui.plot import ExpressPlot
    if shap_bool:
        import shap
        shap.getjs()        

    def gc(x):
        return get_color(x, min_value=min_value, max_value=max_value, colorscale=colorscale, format='hex')

    html = ''

    if shap_bool or plotly_bool:
        def get_radio(id_type: str, description: str):
            def jl_line(name, value):
                return f'document.querySelectorAll(\'.{id}-{name}\')' + \
                    f'.forEach(function(el){{ el.style.display = \'{value}\'; }});'
            states = {'default': ['inherit', 'none', 'none'],
                      'plotly': ['none', 'inherit', 'none'],
                      'shap': ['none', 'none', 'inherit']}
            onclick = jl_line('default', states[id_type][0]) + \
                jl_line('plotly', states[id_type][1]) + \
                jl_line('shap', states[id_type][2])
            return f'<input type="radio" name="{id}" id="{id}-{id_type}" ' + \
                f'{"checked " if id_type == "default" else ""}' + \
                f'onclick="{onclick}"></input><label for="{id}-{id_type}">{description}</label>'

        html += f'<form id="{id}"><div class="radio-group">'
        html += get_radio('default', 'Word highlight')
        if plotly_bool:
            html += get_radio('plotly', 'Bar chart')
        if shap_bool:
            html += get_radio('shap', 'SHAP')
        html += '</div></form>'

    html += original_scores_renderer(content['original_scores']) if 'original_scores' in content else ''

    features, scores, original_scores = content['features'], content['scores'], content['original_scores']

    def format_score(score: float, tol: float = 1e-3) -> str:
        score_value = 'near-zero'
        if score > tol:
            score_value = 'positive'
        elif score < tol:
            score_value = 'negative'
        return f'This token has a {score_value} attribution score of {score}'

    def render_one(tokens_and_scores: list, original_score: float, class_name: Optional[str] = None) -> str:
        scores_dict = dict(tokens_and_scores)
        scores_ = [(token, scores_dict[token] if token in scores_dict else None) for token in features]

        renders = []

        html = ''.join([f'<span class="token" style="background-color: {gc(score) if score else "inherit"};' +
                        (' border-bottom: 3px solid rgba(0, 0, 0, 0.3);' if score is not None else '') + '"' +
                        (f'title="{format_score(score)}"' if score is not None else '') +
                        f'>{token}' +
                        (f'<span class="attribution">{score:.3f}</span>' if score is not None else '') + '</span>'
                        for (token, score) in scores_])
        renders.append(f'<div class="{id}-default" style="display: inherit;">{html}</div>')
        if plotly_bool:
            first_element, second_element = 'token', 'attribution'
            df = pd.DataFrame(scores_dict.items(), columns=[first_element, second_element])
            df = df.sort_values(by=second_element)
            html = ExpressPlot(df, px.bar, x=second_element, y=first_element) \
                .update_traces(marker_color=np.where(df[second_element] >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)) \
                .interactive
            renders.append(f'<div class="{id}-plotly" style="display: none;">{html}</div>')
        if shap_bool:
            html = shap.force_plot(base_value=original_score,
                                   shap_values=np.array([score for _, score in scores_dict.items()]),
                                   feature_names=[token for token, _ in scores_dict.items()],
                                   out_names=class_name).html()
            renders.append(f'<div class="{id}-shap" style="display: none;">{html}</div>')
        return ''.join(renders)

    if isinstance(scores, dict):
        for class_name, score in scores.items():
            html += format_label(class_name, label_name='Class')
            html += render_one(score, original_scores[class_name], class_name)
        return html
    return html + render_one(scores)


@plotly_fallback
def featurelist_renderer(meta: dict,
                         content: dict,
                         first_element: str = 'token',
                         second_element: str = 'frequency',
                         vertical: bool = False,
                         sorted: bool = True,
                         **renderargs) -> str:
    """Render token information/frequency return types."""
    import plotly.express as px
    from genbase.ui.plot import ExpressPlot

    label_name = 'Class'
    if 'callargs' in meta and 'explain_model' in meta['callargs']:
        label_name = 'Predicted class' if meta['callargs']['explain_model'] else 'Ground-truth class'

    def render_one(class_name: str, tokens_and_scores: list):
        html = '' if class_name == 'all' else format_label(class_name, label_name=label_name)
        df = pd.DataFrame(tokens_and_scores, columns=[first_element, second_element])
        if sorted:
            df = df.sort_values(by=second_element)

        x, y = (first_element, second_element) if vertical else (second_element, first_element)
        html += ExpressPlot(df, px.bar, x=x, y=y, color_discrete_sequence=[MAIN_COLOR]).interactive
        return html
    return ''.join(render_one(k, v) for k, v in content.items())


def rules_renderer(meta: dict, content: dict, **renderargs) -> str:
    """Render a set of rules from rule return types."""
    html = original_scores_renderer(content['original_scores']) if 'original_scores' in content else ''

    def render_one(label, rules):
        return format_label(label, label_name='Rules for class') + \
            '<code>' + '\n'.join(rules) + '</code>'

    if isinstance(content['rules'], dict):
        for label, rules in content['rules'].items():
            html += render_one(label, rules)
    else:
        html += render_one(content['label'], content['rules'])
    return html


def frequency_renderer(meta: dict, content: dict, **renderargs) -> str:
    """Render token_frequency return type."""
    return featurelist_renderer(meta,
                                content,
                                first_element='token',
                                second_element='frequency',
                                vertical=False,
                                sorted=True,
                                **renderargs)


def information_renderer(meta: dict, content: dict, **renderargs) -> str:
    """Render token_information return type."""
    return featurelist_renderer(meta,
                                content,
                                first_element='token',
                                second_element='mutual information',
                                vertical=False,
                                sorted=True,
                                **renderargs)


def prototype_renderer(meta: dict, content: dict, **renderargs) -> str:
    """Render prototypes return type."""
    def render_one(instance_type: str, instances) -> str:
        return f'<h4>{instance_type.title()}</h4><p>{format_instances(instances)}</p>'

    def render_class(class_name: Optional[str], instances: dict) -> str:
        html = '' if class_name is None else format_label(class_name, label_name='Class')
        for k, v in instances.items():
            html += render_one(k, v)
        return html

    if all(isinstance(v, list) for v in content.values()):
        return render_class(None, content)
    return ''.join(render_class(class_name, instances) for class_name, instances in content.items())


class Render(BaseRender):
    def __init__(self, *configs):  # noqa: D103
        super().__init__(*configs) 
        self.main_color = MAIN_COLOR
        self.package_link = 'https://text-explainability.readthedocs.io/'
        self.extra_css = """
            .token {
                display: inline-block;
                color: #000;
                padding: 0.8rem 0.7rem;
                margin: 0 0.2rem;
            }

            .token > .attribution {
                color: rgba(0, 0, 0, 0.8);
                vertical-align: super;
                font-size: smaller;
            }

            .token > .attribution::before {
                content: " [";
            }

            .token > .attribution::after {
                content: "]";
            }

            .radio-group input[type=radio] {
                position: absolute;
                visibility: hidden;
                display: none;
            }

            .radio-group label {
                color: #fff !important;
                display: inline-block !important;
                cursor: pointer !important;
                padding: 5px 20px !important;
                opacity: 0.75 !important;
                -webkit-user-select: none;
                -ms-user-select: none;
                user-select: none;
                margin: 0px;
            }

            .radio-group input[type=radio]:checked + label {
                color: #fff !important;
                background: #000 !important;
                border-top: none !important;
            }

            .radio-group label + input[type=radio] + label {
                border-left: solid 2px var(--maincolor);
                border-top: none !important;
            }

            .radio-group {
                display: inline-block !important;
                border-radius: 20px;
                overflow: hidden;
            }
        """.replace('var(--maincolor)', self.main_color)

    def get_renderer(self, meta: dict):  # noqa: D103
        type, subtype, _ = get_meta_descriptors(meta)

        if type == 'global_explanation':
            if 'frequency' in subtype.split('_'):
                return frequency_renderer
            elif 'information' in subtype.split('_'):
                return information_renderer
            elif 'prototypes' in subtype.split('_'):
                return prototype_renderer
        elif type == 'local_explanation':
            if subtype == 'feature_attribution':
                return feature_attribution_renderer
            if subtype in ['rules', 'local_rules']:
                return rules_renderer
        return default_renderer

    def format_title(self, title: str, h: str = 'h1', **renderargs) -> str:  # noqa: D103
        return super().format_title(title, h=h, **renderargs).replace('_', ' ').title()

    def render_subtitle(self, meta: dict, content, **renderargs) -> str:  # noqa: D103
        type, subtype, _ = get_meta_descriptors(meta)
        labelwise = meta['labelwise'] if 'labelwise' in meta else False
        callargs = meta['callargs'] if 'callargs' in meta else ''

        def fmt_method(name: str) -> str:
            name, url = TRANSLATION_DICT[str.lower(name)] if str.lower(name) in TRANSLATION_DICT else (name, '')
            return f'<a href="{url}" target="_blank">{name}</a>' if url else name

        html = []
        if 'method' in meta:
            html.append(f'Explanation generated with method {fmt_method(meta["method"])}.')
        if type == 'global_explanation':
            if callargs:
                if 'explain_model' in callargs:
                    what = 'predictions according to model' if callargs['explain_model'] \
                        else 'ground-truth labels in dataset'
                    how_many = f' (maximized to top-{callargs["k"]})' if 'k' in callargs else ''
                    html.append(f'{subtype.replace("_", " ").capitalize()} of {what}{how_many}.')                
                if 'filter_words' in callargs:
                    tokens = ', '.join(f'"{t}"' for t in callargs['filter_words'])
                    html.append(f'Excluded tokens: {tokens if tokens else "-"}.')
            if labelwise:
                html.append('Grouped by label.')
        return self.format_subtitle('<br>'.join(html)) if html else ''
