# utils/constants.py  —  bản cập nhật
from lxml import etree

SKIP_TAGS = {
    # thẻ rác / trình bày
    "script", "style", "meta", "link", "title",
    "head", "noscript",
    # layout/ảnh
    "img", "figure", "figcaption", "iframe",
    "col", "colgroup", "hr", "br",
    "font", "html", "body"

}

ABS_POS_IMG_CLASS = {"absolute-img", "docs-image"}

def _ancestor_display_none(el) -> bool:
    """True nếu el có cha nào ẩn hoàn toàn (display:none)."""
    while el is not None:
        style = (el.get("style") or "").replace(" ", "").lower()
        if "display:none" in style:
            return True
        el = el.getparent()
    return False


def should_skip(elem: etree._Element) -> bool:
    tag = elem.tag.lower() if isinstance(elem.tag, str) else ""

    # rác cố định
    if tag in SKIP_TAGS:
        return True

    # ancestor bị ẩn ⇒ bỏ toàn bộ subtree
    if _ancestor_display_none(elem):
        return True

    # bỏ nhánh không phải Python, khi parent có cả 2 language
    lang = (elem.get("data-tree-language") or "").lower()
    if lang:
        parent = elem.getparent()
        if parent is not None:
            has_python = any(
                (c.get("data-tree-language") or "").lower() == "python"
                for c in parent
            )
            if has_python and lang != "python":
                return True

    return False


