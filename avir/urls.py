"""Scheme guard for outbound HTTP calls made with ``urllib.request.urlopen``.

``urlopen`` also honours ``file:`` and other handlers, so a coordinator or model
endpoint URL taken from configuration could read local files instead of making
a network request. Every outbound call validates its URL here first.
"""

from __future__ import annotations

from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def require_http_url(url: str) -> str:
    """Return ``url`` unchanged if its scheme is http or https, else raise ValueError."""
    scheme = urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"refusing URL with scheme {scheme!r} (only http/https allowed): {url}")
    return url
