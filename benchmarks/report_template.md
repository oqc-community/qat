#### Performance Regression Tests
{% if not tests %}
No performance regressions detected.
{% else %}
| Test | Main Exec Time (s) | PR Exec Time (s) | Slow-down | Status |
| ------- | ------ | ------ | ------ | ------ |
{% for name, test in tests.items() -%}
| {{name}} | {{ test.min_before }} | {{test.min_after}} | {{test.rel_diff}}x | {% if test.outcome == "success" %} :white_check_mark: {% elif test.outcome == "warning" %} :warning: {% elif test.outcome == "improvement" %} :rocket: {% else %} :x: {% endif %} |
{% endfor %}
{% endif %}