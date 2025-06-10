#!/usr/bin/env python3
"""
Test Dynamic Classification System - UPDATED với Skip Support
Verify that YAML rules work correctly including SKIP patterns
"""

import yaml
from lxml import etree
import random


def load_pattern_rules(rule_file_path: str = "pattern_rules.yaml") -> dict:
    """Load pattern rules from YAML file"""
    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"❌ Rule file {rule_file_path} not found.")
        return {}


def classify_content_type(elem, rules: dict = None) -> str:
    """
    Dynamic content type classifier với comprehensive skip support
    Returns 'SKIP' for all media và unwanted content
    """
    if rules is None:
        rules = load_pattern_rules()

    if not rules:
        return 'documentation_text'

    classes = elem.get('class', '').split()
    attrs = elem.attrib.keys()
    tag = elem.tag.lower()

    # === PRIORITY 1: Check SKIP patterns first ===
    if 'skip_content' in rules:
        for rule in rules['skip_content']:
            # Class-based skip
            if 'class' in rule and rule['class'] in classes:
                return 'SKIP'

            # Tag-based skip (img, video, script, etc.)
            if 'tag' in rule and rule['tag'] == tag:
                return 'SKIP'

            # Attribute-based skip
            if 'attr' in rule and rule['attr'] in attrs:
                return 'SKIP'

    # === PRIORITY 2: Content classification ===
    for content_type, type_rules in rules.items():
        if content_type == 'skip_content':
            continue

        for rule in type_rules:
            if 'class' in rule and rule['class'] in classes:
                return content_type
            if 'attr' in rule and rule['attr'] in attrs:
                return content_type
            if 'tag' in rule and rule['tag'] == tag:
                return content_type

    # === PRIORITY 3: Default ===
    return 'documentation_text'


def test_classification_with_sample_html():
    """Test classification với sample HTML elements including media"""

    print("🧪 Testing Dynamic Classification System với Skip Support")
    print("=" * 60)

    # Load rules
    rules = load_pattern_rules()
    print(f"📁 Loaded rules for: {list(rules.keys())}")

    # Show skip rules if available
    if 'skip_content' in rules:
        skip_count = len(rules['skip_content'])
        print(f"🚫 Skip rules loaded: {skip_count} patterns")
        # Show first few skip rules
        skip_samples = rules['skip_content'][:5]
        print(f"   Sample skip rules: {skip_samples}")

    # *** UPDATED TEST CASES với media content ***
    test_cases = [
        # Code content
        '<div class="csharp">public class Algorithm</div>',
        '<div class="python">def initialize(self):</div>',
        '<div class="code-snippet">code here</div>',

        # Tables
        '<table class="qc-table">Data table</table>',

        # Navigation
        '<div class="toc-h4">Table of contents item</div>',
        '<div class="page-heading">Section heading</div>',

        # API Reference
        '<div class="section-example-container">Example code here</div>',

        # *** MEDIA CONTENT (should be SKIPPED) ***
        '<img class="cover-icon" src="image.png">',
        '<video class="tutorial-video" src="video.mp4">',
        '<div class="cover-content">Cover page content</div>',
        '<script>alert("hello")</script>',
        '<style>.class { color: red; }</style>',
        '<canvas class="chart">Chart here</canvas>',

        # Unknown content
        '<div class="unknown-class">Unknown content</div>',
        '<span>Plain text</span>'
    ]

    print(f"\n🔍 Classification Results ({len(test_cases)} test cases):")
    print("-" * 60)

    skip_count = 0
    content_count = 0

    for i, html_str in enumerate(test_cases, 1):
        try:
            # Parse HTML element
            elem = etree.fromstring(html_str)

            # Classify
            content_type = classify_content_type(elem, rules)

            # Count skip vs content
            if content_type == 'SKIP':
                skip_count += 1
                status_icon = "🚫"
            else:
                content_count += 1
                status_icon = "✅"

            # Display result
            classes = elem.get('class', 'no-class')
            tag = elem.tag
            print(f"{status_icon} {i:2d}. <{tag} class='{classes}'> → {content_type}")

        except etree.XMLSyntaxError:
            print(f"❌ {i:2d}. Failed to parse: {html_str[:50]}...")

    print(f"\n📊 Test Summary:")
    print(f"   ✅ Content elements: {content_count}")
    print(f"   🚫 Skipped elements: {skip_count}")
    print(f"   📝 Skip ratio: {skip_count / len(test_cases) * 100:.1f}%")

    return rules



def main():
    """Main test function"""
    print("🚀 Dynamic Classification Test Suite với Skip Support")


if __name__ == "__main__":
    main()