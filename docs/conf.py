# Configuration file for the Sphinx documentation builder.
import os
import sys
import subprocess
from importlib import metadata
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, os.path.abspath('..'))

project = 'pyna'
copyright = '2024-2026, Wenyin Wei'
author = 'Wenyin Wei'
try:
    release = metadata.version('pyna-chaos')
except metadata.PackageNotFoundError:
    release = '0.8.27'
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'nbsphinx',
    'autoapi.extension',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Keep scratch examples and research notebooks out of the public docs build.
    # They are listed from the tutorial index without being executed by Sphinx.
    'notebooks/examples/*',
    'notebooks/research/*',
    'notebooks/tutorials/general_dynamics_geometry_patterns.ipynb',
    'notebooks/tutorials/analytic_stellarator_geometry_workflow.ipynb',
    'notebooks/tutorials/RMP_resonance_exec.ipynb',
    'notebooks/validate_chaos.ipynb',
]

# HTML theme
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['language-switcher.js']
html_copy_source = False
html_show_sourcelink = False
html_title = 'pyna dynamics toolkit'

html_theme_options = {
    "logo": {
        "text": "pyna",
    },
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-language", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "back_to_top_button": True,
    "navigation_with_keys": True,
    "secondary_sidebar_items": ["page-toc"],
    "pygments_light_style": "a11y-high-contrast-light",
    "pygments_dark_style": "a11y-high-contrast-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WenyinWei/pyna",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# ? MyST parser settings ?
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'dollarmath',
    'html_admonition',
    'html_image',
]

# ? Napoleon settings (Google/NumPy docstrings) ?
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = False

# nbsphinx settings
# Tutorial notebooks are executed locally and committed with their outputs.
# The GitHub Pages workflow only renders those saved outputs, avoiding slow or
# non-reproducible CI reruns of Monte Carlo and field-line tracing tutorials.
nbsphinx_execute = 'never'
nbsphinx_timeout = 300  # seconds per notebook (per-notebook override via metadata "nbsphinx": {"timeout": N})
nbsphinx_allow_errors = False

# ? intersphinx ?
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# ? autodoc ?
autodoc_mock_imports = ['cupy', 'dolfinx', 'ufl', 'petsc4py', 'mpi4py', 'deprecated', 'joblib', 'plotly']
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# AutoAPI provides a complete source-parsed reference for GitHub Pages without
# importing optional runtime backends at documentation build time.
autoapi_type = 'python'
autoapi_dirs = ['../pyna']
autoapi_root = 'en/api/generated'
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_ignore = [
    '*/__pycache__/*',
    '*/build/*',
]

# ? autosectionlabel ?
autosectionlabel_prefix_document = True

# ? copybutton ?
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Suppress duplicate-object warnings from existing source docstrings
suppress_warnings = [
    'autosectionlabel.*',
    'ref.duplicate',
    'ref.python',
    'docutils',
    'toc.not_included',
    'toc.no_title',
    'autoapi.python_import_resolution',
    'sphinx_autodoc_typehints.forward_reference',
]

nitpicky = False


translation_languages = {
    'en': 'English',
    'zh': '中文',
    'ja': '日本語',
    'ko': '한국어',
    'de': 'Deutsch',
    'fr': 'Français',
    'ru': 'Русский',
}

translation_badge_text = {
    'zh': {
        'outdated_title': '翻译可能落后',
        'outdated_message': '英文源页更新较新；本页中文内容可能还未完全同步。',
        'missing_title': '中文翻译缺失',
        'missing_message': '该页面还没有中文版本，当前显示英文源页。',
        'link_label': '查看英文版',
    },
    'ja': {
        'outdated_title': '翻訳が古い可能性があります',
        'outdated_message': '英語の原文ページの方が新しいため、この日本語ページはまだ完全には同期されていない可能性があります。',
        'missing_title': '日本語訳はまだありません',
        'missing_message': 'このページにはまだ日本語版がないため、英語の原文を表示しています。',
        'link_label': '英語版を見る',
    },
    'ko': {
        'outdated_title': '번역이 오래되었을 수 있습니다',
        'outdated_message': '영어 원문 페이지가 더 최근에 업데이트되었으므로 이 한국어 페이지가 아직 완전히 동기화되지 않았을 수 있습니다.',
        'missing_title': '한국어 번역 없음',
        'missing_message': '이 페이지의 한국어 번역이 아직 없어 영어 원문을 표시합니다.',
        'link_label': '영어 원문 보기',
    },
    'de': {
        'outdated_title': 'Übersetzung möglicherweise veraltet',
        'outdated_message': 'Die englische Quellseite ist neuer; diese deutsche Seite ist möglicherweise noch nicht vollständig synchronisiert.',
        'missing_title': 'Deutsche Übersetzung fehlt',
        'missing_message': 'Für diese Seite gibt es noch keine deutsche Fassung; angezeigt wird die englische Quelle.',
        'link_label': 'Englische Quelle öffnen',
    },
    'fr': {
        'outdated_title': 'Traduction peut-être obsolète',
        'outdated_message': 'La page source anglaise est plus récente ; cette page française n’est peut-être pas encore entièrement synchronisée.',
        'missing_title': 'Traduction française indisponible',
        'missing_message': 'Cette page n’a pas encore de version française ; la source anglaise est affichée.',
        'link_label': 'Voir la source anglaise',
    },
    'ru': {
        'outdated_title': 'Перевод может быть устаревшим',
        'outdated_message': 'Английская исходная страница новее; эта русская страница может быть еще не полностью синхронизирована.',
        'missing_title': 'Русский перевод отсутствует',
        'missing_message': 'Для этой страницы пока нет русской версии; показан английский источник.',
        'link_label': 'Открыть английский источник',
    },
    'default': {
        'outdated_title': 'Translation may be outdated',
        'outdated_message': 'The English source page is newer than this translation.',
        'missing_title': 'Translation unavailable',
        'missing_message': 'This page is currently falling back to the English source.',
        'link_label': 'View English source',
    },
}


def _page_language(pagename):
    parts = pagename.split('/')
    if parts and parts[0] in translation_languages:
        return parts[0]
    if parts and parts[0] == 'notebooks':
        return 'en'
    return None


def _english_counterpart(pagename, lang):
    parts = pagename.split('/')
    if not parts:
        return 'en/index'
    if lang == 'en':
        return pagename
    return '/'.join(['en'] + parts[1:])


def _rst_source_path(srcdir, pagename):
    return Path(srcdir).joinpath(*pagename.split('/')).with_suffix('.rst')


@lru_cache(maxsize=None)
def _git_timestamp(srcdir, source_path):
    srcdir_path = Path(srcdir)
    path = Path(source_path)
    try:
        relpath = path.relative_to(srcdir_path)
    except ValueError:
        relpath = path

    dirty = subprocess.run(
        ['git', '-C', str(srcdir_path), 'status', '--porcelain', '--', str(relpath)],
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.stdout.strip():
        return path.stat().st_mtime

    result = subprocess.run(
        ['git', '-C', str(srcdir_path), 'log', '-1', '--format=%ct', '--', str(relpath)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip().splitlines()[0])
    return path.stat().st_mtime if path.exists() else None


def _translation_badge_for_page(app, pagename):
    lang = _page_language(pagename)
    if not lang or lang == 'en':
        return None

    source = _rst_source_path(app.srcdir, pagename)
    english_page = _english_counterpart(pagename, lang)
    english_source = _rst_source_path(app.srcdir, english_page)
    if not english_source.exists():
        return None

    source_time = _git_timestamp(app.srcdir, str(source)) if source.exists() else None
    english_time = _git_timestamp(app.srcdir, str(english_source))
    if source_time is None or english_time is None or english_time <= source_time:
        return None

    text = translation_badge_text.get(lang, translation_badge_text['default'])
    return {
        'status': 'outdated',
        'title': text['outdated_title'],
        'message': text['outdated_message'],
        'link_label': text['link_label'],
        'english_page': english_page,
    }


def _add_translation_badge_context(app, pagename, templatename, context, doctree):
    context['pyna_translation_languages'] = translation_languages
    context['pyna_current_language'] = _page_language(pagename) or 'root'
    context['pyna_translation_badge'] = _translation_badge_for_page(app, pagename)


def _disable_nbsphinx_notebook_copy(app):
    """Render notebooks as HTML without publishing raw .ipynb sources."""
    if app.builder.format == 'html' and hasattr(app.env, 'nbsphinx_notebooks'):
        app.env.nbsphinx_notebooks = {}
    return []


def _remove_html_source_maps(app, exception):
    """Keep published pages compact and free of third-party source-map payloads."""
    if exception is not None or app.builder.format != 'html':
        return
    for root, _dirs, files in os.walk(app.builder.outdir):
        for filename in files:
            path = os.path.join(root, filename)
            if filename.endswith(('.css', '.js')):
                try:
                    with open(path, encoding='utf-8') as stream:
                        lines = stream.readlines()
                except UnicodeDecodeError:
                    lines = []
                filtered = [line for line in lines if 'sourceMappingURL=' not in line]
                if filtered != lines:
                    with open(path, 'w', encoding='utf-8') as stream:
                        stream.writelines(filtered)
            if filename.endswith('.map'):
                os.remove(path)


def setup(app):
    # nbsphinx copies executed notebooks during html-collect-pages.  The pages
    # already contain rendered outputs, and publishing raw JSON can expose large
    # base64 payloads, so keep GitHub Pages to HTML/assets only.
    app.connect('html-page-context', _add_translation_badge_context)
    app.connect('html-collect-pages', _disable_nbsphinx_notebook_copy, priority=400)
    app.connect('build-finished', _remove_html_source_maps)
