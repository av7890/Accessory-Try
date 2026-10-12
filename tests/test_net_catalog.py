import io
import socket

import numpy as np
import pytest
from PIL import Image

from tryon import catalog
from tryon.net import FetchError, safe_decode, validate_url


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a.png", "gopher://x", "javascript:alert(1)",
    "http://127.0.0.1/a.png", "http://localhost/a.png", "http://[::1]/a.png",
    "http://10.0.0.5/a.png", "http://192.168.1.1/a.png", "http://169.254.169.254/latest/meta-data",
    "http://0.0.0.0/a.png", "http://[::ffff:127.0.0.1]/a.png",
    "http://user:pw@8.8.8.8/a.png", "http://8.8.8.8:22/a.png", "http://8.8.8.8:8080/a.png", "", "http://",
])
def test_blocked_urls(url):
    with pytest.raises(FetchError):
        validate_url(url)


def test_public_ip_allowed_and_pinned():
    scheme, host, port, path, ip = validate_url("https://8.8.8.8/x/y.png?a=1")
    assert (scheme, port, ip, path) == ("https", 443, "8.8.8.8", "/x/y.png?a=1")


def test_dns_name_resolving_to_private_is_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))])
    with pytest.raises(FetchError):
        validate_url("http://evil.example/a.png")


def test_dns_rebinding_mixed_answers_blocked(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(FetchError):
        validate_url("http://rebind.example/a.png")


def _png(w=64, h=64, fmt="PNG"):
    b = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(b, fmt)
    return b.getvalue()


def test_safe_decode_accepts_images_rejects_junk():
    assert safe_decode(_png()).shape[:2] == (64, 64)
    for bad in (b"", b"not an image", b"<svg xmlns='http://www.w3.org/2000/svg'/>", _png(8, 8)):
        with pytest.raises(ValueError):
            safe_decode(bad)


def test_safe_decode_blocks_huge_dimensions(monkeypatch):
    import tryon.net as net
    monkeypatch.setattr(net, "MAX_PIXELS", 1000)
    with pytest.raises(ValueError):
        safe_decode(_png(64, 64))


def test_catalog_roundtrip_and_traversal(tmp_path):
    s = np.zeros((20, 20, 4), np.uint8)
    s[5:15, 5:15] = (1, 2, 3, 255)
    i = catalog.save_item("My Ring #1", "Ring", s, tmp_path)
    assert catalog.list_items(tmp_path)[0]["type"] == "Ring"
    assert catalog.load_item(i, tmp_path).shape == (20, 20, 4)
    assert catalog.save_item("My Ring #1", "Ring", s, tmp_path) != i        # unique ids
    for evil in ("../index", "..\\x", "a/b", "", "A" * 80):
        assert catalog.load_item(evil, tmp_path) is None
    with pytest.raises(ValueError):
        catalog.save_item("x", "Hat", s, tmp_path)
    catalog.delete_item(i, tmp_path)
    assert catalog.load_item(i, tmp_path) is None
    catalog.delete_item("../../etc/passwd", tmp_path)                        # silently ignored
