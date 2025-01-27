## {{ package }} `{{ status }}`
https://github.com/{{repo_name}}

{% if summaries.unit -%}
#### Unit Tests
| Operating System | Python Version | Result | Passed ✅ | Failed ❌ | Errors ❗ | Skipped ↩️ | Notes |
| ------| ------| ------ | ------ | ------ | ------ | ------- | ------ |
{% for summary in summaries.unit -%}
| {{ summary.os }} | {{ summary.python }} | {% if summary.outcome == "success" -%} ✅ `Pass` {% else -%} ❌ `Fail` {% endif -%} | {{ summary.passed }} | {{ summary.failures }} | {{ summary.errors }} | {{ summary.skipped }} | {{ summary.notes }} |
{% endfor -%}
{% endif -%}

{% if summaries.integration -%}
#### Integration Tests

| Service | Result | Passed ✅ | Failed ❌ | Errors ❗ | Skipped ↩️ | Notes |
| ------- | ------ | ------ | ------ | ------ | ------- | ------ |
{% for summary in summaries.integration -%}
| {{summary.service}} | {% if summary.outcome == "success" -%} ✅ `Pass` {% else -%} ❌ `Fail` {% endif -%} | {{ summary.passed }} | {{ summary.failures }} | {{ summary.errors }} | {{ summary.skipped }} | {{ summary.notes }} |
{% else -%}
No integration tests defined
{% endfor -%}
{% endif -%}

{% if details.failure -%}
#### Failures

{% for case in details.failure -%}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
<br/>  

{% endfor -%}
{% endif -%}

{% if details.error -%}
#### Errors

{% for case in details.error -%}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
<br/>  

{% endfor -%}
{% endif -%}

{% if details.skipped and show_skipped -%}
#### Skipped

{% for case in details.skipped -%}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
<br/>  

{% endfor -%}
{% endif -%}