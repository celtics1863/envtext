def escape_html(text):
    """Replace <, >, &, " with their HTML encoded representation. Intended to
    prevent HTML errors in rendered displaCy markup.
    text (str): The original text.
    RETURNS (str): Equivalent text to be safely used within HTML.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text

def minify_html(html: str) -> str:
    """Perform a template-specific, rudimentary HTML minification for displaCy.
    Disclaimer: NOT a general-purpose solution, only removes indentation and
    newlines.
    html (str): Markup to minify.
    RETURNS (str): "Minified" HTML.
    """
    return html.strip().replace("    ", "").replace("\n", "")
