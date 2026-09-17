"""We store what we render, not what upstream serves.

Commons' ``Special:FilePath`` hands back the full-resolution master — Katie Couric's portrait
is 16 MB — and even a thumbnailed rendition arrives far larger than the 176pt square the person
card draws. Every one of those bytes is then stored in the corpus, carried in every backup and
served on every card view, forever.

:func:`_downscale_image` is the single choke point, applied at BOTH provider return sites
(person photos and org logos). The properties that matter:

1. **Oversized images shrink**, and shrink to valid bytes of the SAME format — the sidecar
   records ``ext`` and ``/persons/{id}/photo`` serves by it, so a JPEG written into a ``.png``
   would break the route.
2. **Already-small images are returned untouched** — no needless generation loss.
3. **A decode failure costs us nothing.** A photo we cannot resize is still a photo; the bytes
   have already been sniffed and size-capped by the caller.
4. **The verdict is DIMENSIONS, not bytes.** A 1600px master that happens to compress smaller
   than its own 512px rendition (real: PNG, structured content) is still a 1600px master we
   would store, back up and serve. Byte size is not the test.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _downscale_image,
    _IMAGE_MAX_EDGE,
    _image_sniff_ok,
)

PIL = pytest.importorskip("PIL.Image", reason="pillow is a declared runtime dep")


def _img(w: int, h: int, ext: str, mode: str = "RGB") -> bytes:
    from PIL import Image

    # Incompressible noise, not flat colour and not an arithmetic ramp: both compress to
    # almost nothing, and a ramp actually compresses BETTER at full size than downscaled,
    # which is how the byte-size guard this module now rejects got written in the first place.
    raw = hashlib.sha256(b"seed").digest()
    while len(raw) < w * h * len(mode):
        raw += hashlib.sha256(raw[-32:]).digest()
    im = Image.frombytes(mode, (w, h), raw[: w * h * len(mode)])
    buf = io.BytesIO()
    im.save(buf, {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}[ext])
    return buf.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


class TestOversizedImagesShrink:
    @pytest.mark.parametrize("ext", ["jpg", "png", "webp"])
    def test_longest_edge_is_capped_and_format_survives(self, ext):
        original = _img(1600, 1200, ext)

        out = _downscale_image(original, ext)

        assert max(_size(out)) <= _IMAGE_MAX_EDGE
        assert len(out) < len(original), "the point of the exercise is fewer bytes"
        assert _image_sniff_ok(ext, out), (
            "format must not change: the sidecar records ext and /persons/{id}/photo serves "
            "the file by that extension"
        )

    def test_aspect_ratio_is_preserved(self):
        """Downscale, never crop — a crop box here would be deciding where a face sits."""
        out = _downscale_image(_img(1600, 800, "jpg"), "jpg")

        w, h = _size(out)
        assert w == _IMAGE_MAX_EDGE and h == _IMAGE_MAX_EDGE // 2

    def test_a_portrait_orientation_image_caps_on_height(self):
        out = _downscale_image(_img(600, 1500, "jpg"), "jpg")

        w, h = _size(out)
        assert h == _IMAGE_MAX_EDGE, "the LONGEST edge is the one capped"
        assert w < _IMAGE_MAX_EDGE


class TestSmallImagesAreLeftAlone:
    def test_an_image_within_the_cap_is_returned_byte_identical(self):
        """No re-encode means no generation loss on a photo that was already fine."""
        original = _img(300, 200, "jpg")

        assert _downscale_image(original, "jpg") is original

    def test_an_image_exactly_at_the_cap_is_untouched(self):
        original = _img(_IMAGE_MAX_EDGE, _IMAGE_MAX_EDGE, "png")

        assert _downscale_image(original, "png") is original


class TestFailureNeverCostsThePhoto:
    def test_undecodable_bytes_are_passed_through_unchanged(self):
        """The caller already sniffed and size-capped these bytes; dropping them is worse."""
        garbage = b"\xff\xd8\xff" + b"not actually a jpeg" * 50

        assert _downscale_image(garbage, "jpg") == garbage

    def test_empty_bytes_do_not_raise(self):
        assert _downscale_image(b"", "jpg") == b""

    def test_invalid_encoder_output_falls_back_to_the_original(self, monkeypatch):
        """If the re-encode is not a usable image of the declared type, keep what we had."""
        from PIL import Image

        original = _img(1600, 1200, "png")
        monkeypatch.setattr(Image.Image, "save", lambda self, fp, *a, **kw: fp.write(b"junk"))

        assert _downscale_image(original, "png") == original


class TestDimensionsBeatBytes:
    def test_a_larger_re_encode_is_STILL_preferred_when_it_is_smaller_on_screen(self):
        """The regression this guards: PNG noise re-encodes bigger, and we keep it anyway.

        Measured while writing this: a 1600x1200 ramp-pattern PNG is 48089 B, its 512x384
        rendition 51255 B. Rejecting on bytes would have stored the 1600px frame forever —
        the exact thing the operator asked us to stop doing.
        """
        from PIL import Image

        # A gradient: highly structured, so PNG filters shrink the BIG one best.
        big = Image.frombytes(
            "RGB", (1600, 1200), bytes((i * 7919) % 256 for i in range(1600 * 1200 * 3))
        )
        buf = io.BytesIO()
        big.save(buf, "PNG")
        original = buf.getvalue()

        out = _downscale_image(original, "png")

        assert max(_size(out)) <= _IMAGE_MAX_EDGE, "dimensions are the contract"
        assert len(out) > len(original), (
            "this fixture really is the pathological case — if it ever stops being bigger, "
            "the test has lost its teeth"
        )
