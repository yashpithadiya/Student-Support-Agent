"""Regression guards for layout bugs that hid content (page area not scrolling; login icon colours)."""
import os, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def test_admin_and_student_grid_rows_are_bounded_so_the_stage_can_scroll():
    for name in ("admin.html", "index.html"):
        css = (ROOT / "templates" / name).read_text(encoding="utf-8")
        app = re.search(r"\.app\{display:grid;[^}]*\}", css).group(0)
        assert "grid-template-rows:minmax(0,1fr)" in app and "height:100vh" in app, name
        assert re.search(r"\.main\{[^}]*min-height:0", css) and re.search(r"\.stage\{[^}]*overflow(-y)?:auto", css), name


def test_login_feature_icons_use_four_distinct_colours():
    html = (ROOT / "templates" / "login.html").read_text(encoding="utf-8")
    assert re.findall(r'<div class="feat"><i class="(c\d)"', html) == ["c1", "c2", "c3", "c4"]
    for c in ("#1f6bff", "#12a06a", "#7a4dff", "#e89b0c"):
        assert c in html


def test_headlines_use_the_approved_wording():
    login = (ROOT / "templates" / "login.html").read_text(encoding="utf-8")
    assert "Your Campus.<br>Your Questions.<span>Your Answers.</span>" in login and "smoother" not in login
    page = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'Hi {{ name }},</span>Your campus. Your space.</h1>' in page and "Ready to make" not in page
