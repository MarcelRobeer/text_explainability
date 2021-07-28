"""Support for i18n internationalization."""

import os
import i18n

from typing import List

i18n.load_path.append(os.path.join(os.path.dirname(__file__), 'locale'))
i18n.set('locale', 'nl')
i18n.set('fallback', 'en')


def translate_string(id: str) -> str:
    """Get a string based on `locale`, as defined in the './locale' folder.

    Args:
        id (str): Identifier of string in `lang.{locale}.yml` file.

    Returns:
        str: String corresponding to locale.
    """
    return i18n.t(f'lang.{id}')


def translate_list(id: str, sep: str = ';') -> List[str]:
    """Get a list based on `locale`, as defined in the './locale' folder.

    Args:
        id (str): Identifier of list in `lang.{locale}.yml` file.
        sep (str, optional): Separator to split elements of list. Defaults to ';'.

    Returns:
        List[str]: List corresponding to locale.
    """
    return i18n.t(f'lang.{id}').split(sep)


def set_locale(locale: str) -> None:
    """Set current locale (choose from `en`, `nl`).

    Args:
        locale (str): Locale to change to.
    """
    return i18n.set('locale', locale)


def get_locale() -> str:
    """Get current locale.

    Returns:
        str: Current locale.
    """
    return i18n.get('locale')
