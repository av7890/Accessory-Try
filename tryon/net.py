"""Hardened image fetching / decoding for untrusted input (URLs and uploads).

* only http/https on ports 80/443, no credentials in the URL
* every resolved address must be a public one (blocks loopback, private, link-local,
  cloud-metadata 169.254.169.254, multicast, reserved) -- checked again on every redirect
* the connection is pinned to the validated IP, so DNS rebinding cannot swap it afterwards
* hard caps on body size, redirects, time and decoded pixel count
* errors raised here carry fixed, user-safe messages (no internal detail leaks)
"""
import http.client
import io
import ipaddress
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit

import cv2
import numpy as np
from PIL import Image

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_REDIRECTS = 3
TIMEOUT = 8.0
ALLOWED_PORTS = {80: "http", 443: "https"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class FetchError(Exception):
    """Message is safe to show to the user."""


def _is_public(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_url(url: str):
    """Return (scheme, host, port, path_with_query, pinned_ip) or raise FetchError."""
    if not isinstance(url, str) or len(url) > 2048:
        raise FetchError("Invalid URL.")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise FetchError("Only http and https URLs are allowed.")
    if not parts.hostname or "@" in parts.netloc:
        raise FetchError("Invalid URL.")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise FetchError("Invalid URL.")
    if port not in ALLOWED_PORTS:
        raise FetchError("Only ports 80 and 443 are allowed.")
    try:
        infos = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        raise FetchError("Could not resolve the host.")
    ips = [i[4][0] for i in infos]
    if not ips or not all(_is_public(ip) for ip in ips):
        raise FetchError("This address is not allowed.")
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return parts.scheme, parts.hostname, port, path, ips[0]


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        raw = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def safe_fetch(url: str, max_bytes: int = MAX_BYTES) -> bytes:
    deadline = time.monotonic() + TIMEOUT * 2
    for _ in range(MAX_REDIRECTS + 1):
        scheme, host, port, path, ip = validate_url(url)
        cls = _PinnedHTTPS if scheme == "https" else _PinnedHTTP
        conn = cls(host, port, ip, TIMEOUT)
        try:
            conn.request("GET", path, headers={"Host": host, "User-Agent": "Mozilla/5.0",
                                               "Accept": "image/*", "Accept-Encoding": "identity"})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location")
                if not loc:
                    raise FetchError("Bad redirect.")
                url = urljoin(url, loc)
                continue
            if resp.status != 200:
                raise FetchError("The server did not return the image.")
            if not (resp.getheader("Content-Type") or "").lower().startswith("image/"):
                raise FetchError("The URL is not an image.")
            if int(resp.getheader("Content-Length") or 0) > max_bytes:
                raise FetchError("Image is too large.")
            buf = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    return bytes(buf)
                buf += chunk
                if len(buf) > max_bytes:
                    raise FetchError("Image is too large.")
                if time.monotonic() > deadline:
                    raise FetchError("The download took too long.")
        except FetchError:
            raise
        except (OSError, http.client.HTTPException, ValueError):
            raise FetchError("Could not download the image.")
        finally:
            conn.close()
    raise FetchError("Too many redirects.")


def safe_decode(data: bytes, max_bytes: int = MAX_BYTES * 2) -> np.ndarray:
    """Validate header (format, size) with Pillow before the native decoder sees the bytes."""
    if not data or len(data) > max_bytes:
        raise ValueError("Image is empty or too large.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format.")
            w, h = im.size
            if w * h > MAX_PIXELS or min(w, h) < 16:
                raise ValueError("Image dimensions are not supported.")
    except ValueError:
        raise
    except Exception:
        raise ValueError("Not a valid image.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Not a valid image.")
    return img
