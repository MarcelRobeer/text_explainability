"""Extension of `genbase.ui.notebook` for custom rendering of `text_explainability."""

from genbase.ui import get_color
from genbase.ui.notebook import Render as BaseRender
from genbase.ui.notebook import plotly_available


def default_renderer(meta: dict, content: dict, **renderargs) -> str:
    return f'<p>{content}</p>'


def plotly_fallback(function):
    def inner(*args, **kwargs):
        return function(*args, **kwargs) if not plotly_available() else default_renderer(*args, **kwargs)
    return function


def feature_attribution_renderer(meta: dict, content, **renderargs) -> str:
    min_value = renderargs.pop('min_value', -1.0)
    max_value = renderargs.pop('max_value', 1.0)
    colorscale = renderargs.pop('colorscale', 'RdBu')

    def gc(x):
        return get_color(x, min_value=min_value, max_value=max_value, colorscale=colorscale, format='hex')

    features, scores = content['features'], content['scores']

    def render_one(tokens_and_scores: list):
        scores_dict = dict(tokens_and_scores)
        scores_ = [(token, scores_dict[token] if token in scores_dict else None) for token in features]
        return ''.join([f'<span class="token" style="background-color: {gc(score) if score else "inherit"}">{token}' +
                        (f' [{score:.3f}]' if score is not None else '') + '</span>'
                        for (token, score) in scores_])

    if isinstance(scores, dict):
        html = ''
        for class_name, score in scores.items():
            html += f'<h3>{class_name}</h3>'
            html += render_one(score)
        return html
    return render_one(scores)


@plotly_fallback
def frequency_renderer(meta: dict, content: dict, **renderargs) -> str:
    print('Frequency')
    return f'{content}'


class Render(BaseRender):
    def __init__(self, *configs):
        super().__init__(*configs) 
        self.main_color = '#1976D2'
        self.package_link = 'https://git.io/text_explainability'
        self.extra_css = """
            .token {
                display: inline-block;
                color: #000;
                padding: 1.5rem 1rem;
                margin: 0;
            }
        """

    def get_renderer(self, meta: dict):
        def fmt(x):
            return str(x).strip().lower().replace(' ', '_')

        def get_from_meta(key: str) -> str:
            return fmt(meta[key]) if key in meta else ''

        type = get_from_meta('type')
        subtype = get_from_meta('subtype')
        method = get_from_meta('method')  # noqa: F841

        if type == 'global_explanation':
            if 'frequency' in subtype.split('_'):
                return frequency_renderer
        elif type == 'local_explanation':
            if subtype == 'feature_attribution':
                return feature_attribution_renderer
        return default_renderer

    def format_title(self, title: str, h: str = 'h1', **renderargs) -> str:
        return super().format_title(title, h=h, **renderargs).replace('_', ' ').title()

    def render_subtitle(self, meta: dict, content, **renderargs) -> str:
        def fmt_method(name: str) -> str:
            translation_dict = {'lime': ('LIME', 'https://christophm.github.io/interpretable-ml-book/lime.html'),
                                'shap': ('SHAP', 'https://christophm.github.io/interpretable-ml-book/shap.html'),
                                'kernelshap': ('KernelSHAP', '')}
            name, url = translation_dict.pop(str.lower(name), (name, ''))
            return f'<a href="{url}" target="_blank">{name}</a>' if url else name

        if 'method' in meta:
            return self.format_subtitle(f'Explanation generated with method {fmt_method(meta["method"])}.')
        return ''

    def render_content(self, meta: dict, content, **renderargs) -> str:
        renderer = self.get_renderer(meta)
        return renderer(meta, content, **renderargs)
