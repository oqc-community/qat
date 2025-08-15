## {{ package }} Pytest Report `{{ status }}`
https://github.com/{{repo_name}}

{% if summaries.unit -%}
#### Unit Tests
| Operating System | Python Version | Codebase | Result | Passed ✅ | Failed ❌ | Errors ❗ | Skipped ↩️ | Notes |
| ------| ------| ------ | ------ | ----- |------ | ------ | ------- | ------ |
{% for summary in summaries.unit -%}
| {{ summary.os }} | {{ summary.python }} | {{ summary.codebase }} |{% if summary.outcome == "success" -%} ✅ `Pass` {% else -%} ❌ `Fail` {% endif -%} | {{ summary.passed }} | {{ summary.failures }} | {{ summary.errors }} | {{ summary.skipped }} | {{ summary.notes }} |
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

{% set max_cases = 10 -%}

{% if details.failure -%}
<details>

<Summary>Failure Report</Summary>

#### Failures


{% if details.failure|length > max_cases -%}
> **⚠️ Warning:** Only the first {{ max_cases }} failures are shown. See the full report for more details.
{% endif -%}
{% for case in details.failure[:max_cases] %}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
{% if verbose -%}
```
{{ case.verbose }}
```
{% endif -%}
<br/>  

{% endfor -%}
</details>
{% endif -%}

{% if details.error -%}
<details>

<Summary>Error Report</Summary>

#### Errors


{% if details.error|length > max_cases -%}
> **⚠️ Warning:** Only the first {{ max_cases }} errors are shown. See the full report for more details.
{% endif -%}
{% for case in details.error[:max_cases] %}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
{% if verbose -%}
```
{{ case.verbose }}
```
{% endif -%}
<br/>  

{% endfor -%}
</details>
{% endif -%}

{% if details.skipped and show_skipped -%}
<details>

<Summary>Skipped Report</Summary>

#### Skipped


{% if details.skipped|length > max_cases -%}
> **⚠️ Warning:** Only the first {{ max_cases }} skipped tests are shown. See the full report for more details.
{% endif -%}
{% for case in details.skipped[:max_cases] %}

**{{case.classname}}::{{case.name}} (os: {{case.os}} python: {{case.python}})**  
```python
{{case.message}}
```
<br/>  

{% endfor -%}
</details>
{% endif -%}