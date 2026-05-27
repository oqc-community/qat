# Skill: Create or update a Jira ticket

## Purpose

Create and update Jira tickets in the `COMPILER` project using `acli`:

- **Create**: `acli jira workitem create --from-json <file>`
- **Update**: `acli jira workitem edit --key <KEY> --description-file <file> --yes`

## Inputs

- Ticket summary.
- Description / scope text.
- Acceptance criteria bullets.
- Definition of done bullets.
- (Optional) Labels, parent epic, issue type overrides.

## Formatting

Jira Cloud (accessed via `acli`) uses **Atlassian Document Format (ADF)** — it does **not** render
wiki markup (`h2.`, `*bold*`, `-` bullets, `{{code}}`). Always write descriptions as ADF JSON.

Key ADF node types:

| Intent      | ADF type / attrs                                                                             |
| ----------- | -------------------------------------------------------------------------------------------- |
| Heading     | `{"type":"heading","attrs":{"level":2},"content":[...]}`                                     |
| Paragraph   | `{"type":"paragraph","content":[...]}`                                                       |
| Bullet list | `{"type":"bulletList","content":[{"type":"listItem","content":[{"type":"paragraph",...}]}]}` |
| Plain text  | `{"type":"text","text":"..."}`                                                               |
| Bold text   | `{"type":"text","text":"...","marks":[{"type":"strong"}]}`                                   |
| Inline code | `{"type":"text","text":"...","marks":[{"type": "code"}]}`                                    |

Wrap the full content array in the ADF document envelope:

```json
{"version":1,"type":"doc","content":[...]}
```

For **creates**, embed the parsed ADF object (not a string) in the `--from-json` payload under
`description`. For **updates**, write the ADF JSON to a temp file and pass with
`--description-file <file>`.

## Defaults

Always-on defaults (apply to every ticket unless the user overrides):

- `project.key`: `COMPILER`
- `issuetype.name`: `Task`
- `customfield_10001.id`: `b2ce056a-4429-43d3-a276-b80159c4f4c1` (team field)

Tech-debt defaults (only apply when the ticket is explicitly for tech-debt work; otherwise omit or
ask the user):

- `parent.key`: `COMPILER-912` (current tech-debt epic)
- `labels`: `["TechDebt"]`

## Procedure

1. Build description using this section structure (adjust sections to suit):
   - `Summary` — one paragraph scope statement
   - `Implementation guidance` — brief guidance on what to change and where
   - `Acceptance Criteria` — bullet list
   - `Definition of Done` — bullet list
1. Encode the description as ADF JSON (see **Formatting** above). Write to a temp file.
1. For **creates**, build a JSON payload using values from **Defaults** above and run
   `acli jira workitem create --from-json <file>`.
1. For **updates**, run `acli jira workitem edit --key <KEY> --description-file <file> --yes`.
1. Return the ticket key and URL.
1. If originating from a PR thread, post the URL to that thread and resolve when instructed.

## Output format

- Ticket key and URL.
- Effective parent epic and label values (for creates).
- Follow-up status for linked PR thread, if any.

## Guardrails

- Use `acli`; do not use alternative Jira CLIs.
- **Never** use wiki markup (`h2.`, `*text*`, `{{code}}`); always use ADF JSON.
- Keep description concise and factual.
- Do not include PR thread IDs in the ticket description body.
