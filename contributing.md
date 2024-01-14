# Contributing to QAT

First and foremost, thank you for considering contributing to QAT - an open-source, community maintained project.
We are very excited to have you and your ideas working to improve both the codebase and the community.
To begin, read the project overview and guidelines to help you quickly familiarise yourself with the project and the way
we do things.

The areas you could contribute to QAT include:

- Submitting bug reports and feature requests
- Contributing to the codebase (e.g. new features, improving the test suite etc.)
- Documentation
- Helping to answer questions which are asked in the discussion forum

All contributions to QAT are governed by our
[code of conduct](https://github.com/oqc-community/qat/blob/main/CODE_OF_CONDUCT.rst).

## Installation

To get started, see the Building from Source section of the
[README](https://github.com/oqc-community/qat/blob/main/README.rst).

## Pull Requests

To begin, fork this repository, make changes in your own fork, and then submit a pull request (PR).
All new code should have associated unit tests that validate implemented features and ensure the proper functionality of
the code.

A PR must receive maintainer approval before it can be merged. We aim to review significant contributions within two weeks
and minor changes within a few days, but we cannot guarantee this will always be the case. We expect reviews to be carried
out respectfully and that a commitment to maintain a standard of excellence in the code base will be paramount.
If a pull request has become stale and inactive, the project team may choose to close the PR.
As an open source project, your patience is appreciated when waiting for PR reviews.

## Your First Contribution

Now you are all set to start contributing to QAT!
If you are unsure where to start, please check out our 'good first issue' and 'help wanted' issues.
The 'good first issue' tasks will require only small changes to the code base and are intended to provide you with a smooth
accessible learning curve whilst familiarising yourself with QAT.
The 'help wanted' issues will require greater interaction/modifications to the codebase and are intended for more
experienced contributors.

## First-Timer Help

If you're entirely new to contributing to open source projects, some of the following helpful links may be useful to you:

- For an overview of how to take part in an open-source project, please read
  [this](https://www.freecodecamp.org/news/how-to-contribute-to-open-source-projects-beginners-guide/)
- Need some help with how to PR? Please visit this
  [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
- Never had your code reviewed before? For information about how this process works, please click
  [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)

Please ask for help when you need it- after all, we were all beginners at first.

## How to Report a Bug

### Security Vulnerabilities

If you believe you have found a security vulnerability, do not open an issue or report it publicly.
Instead, please email the security team at qat_security. Once a security vulnerability has
been reported, you should receive an acknowledgement in two working days.
If you suspect you have found a security issue but are unsure, please email the project team in any case.


### Opening an Issue

If you believe you've found an issue and would like to raise it:

- First check to see if your issue has already been fixed. If so, download the version with the fix and try again.
- Has it been raised before by someone else? If an issue is already open, see if you can add additional information to help diagnose or recreate the issue.
- Otherwise open an issue. We have an issue template set up so please use that
  to guide you and give examples of the information we're looking for. The more precise details you provide the higher chance we'll be able to reproduce and fix the issue in
  a decent timeframe.

## New Features

Before trying to add a new feature, open an [issue](https://github.com/oqc-community/qat/issues) or start a [discussion](https://github.com/oqc-community/qat/discussions) with an overview of what you'd like to see.
This is to save disappointment as some features we will refuse because purely it doesn't fit with QAT as a whole or where we're currently going.

Once a new feature has approval of the team as a whole you can start work on it and we'll support you where possible. If you do not want to work on it personally it'll go on the backlog and we'll get to it when possible.

For ideas about where the project is heading, checkout the [issues](https://github.com/oqc-community/qat/issues) list
or the Roadmap in the [README](https://github.com/oqc-community/qat/blob/main/README.rst).

## Additional Information

### Unit Tests

QAT has an existing test suite, which can be found in the tests directory of the code base.
It is our policy that these tests should pass at all times, unless a test is skipped in which case justification should
be provided.
New features should be accompanied by a comprehensive test suite which makes sure that the feature is behaving as
intended, and that future merges do not unintentionally break this functionality.
