import pytest

from text_explainability.internationalization import set_locale, get_locale, translate_string, translate_list

locale = ['nl', 'en']
ids = [i for i in range(len(locale))]

def test_swap_locale():
    set_locale(locale[0])
    assert get_locale() == locale[0]
    set_locale(locale[1])
    assert get_locale() == locale[1]

@pytest.mark.parametrize('id', ids)
def test_translate_string(id):
    set_locale(locale[id])
    assert isinstance(translate_string('str1'), str)

@pytest.mark.parametrize('id', ids)
def test_translate_string_length(id):
    set_locale(locale[id])
    assert len(translate_string('str1')) > 0

@pytest.mark.parametrize('id', ids)
def test_translate_list(id):
    set_locale(locale[id])
    assert isinstance(translate_list('list1'), list)

@pytest.mark.parametrize('id', ids)
def test_translate_list_length(id):
    set_locale(locale[id])
    assert len(translate_list('list1')) > 0
