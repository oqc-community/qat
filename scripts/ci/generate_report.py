import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

NON_PASSED = ["failure", "skipped", "error"]


def get_details(testsuite, os, python, codebase):
    details = {}
    for tag in NON_PASSED:
        details[tag] = []
        for node in testsuite.findall(f"testcase/{tag}/.."):
            output = {
                "classname": node.get("classname"),
                "name": node.get("name"),
                "message": node.find(tag).get("message"),
                "verbose": node.find(tag).text,
                "os": os,
                "python": python,
                "codebase": codebase,
            }
            details[tag].append(output)
    return details


def has_passed(testcase):
    for tag in NON_PASSED:
        if testcase.find(tag) is not None:
            return False
    return True


def split_path(path):
    (os, python, suite, codebase, _) = path.stem.split("_")
    return (os, python, suite, codebase)


def parse_junit(path):
    (os, python, codebase, _) = split_path(path)
    root = ET.parse(path).getroot()
    testsuite = root[0]
    details = get_details(testsuite, os, python, codebase)
    summary = {
        k: int(v)
        for (k, v) in testsuite.attrib.items()
        if k in ["errors", "failures", "skipped", "tests"]
    }
    summary["passed"] = sum(
        has_passed(testcase) for testcase in testsuite.findall("testcase")
    )
    summary["outcome"] = (
        "success" if sum(summary[tag] for tag in ["errors", "failures"]) == 0 else "failure"
    )
    summary["os"] = os
    summary["python"] = python
    summary["codebase"] = codebase
    return summary, details


def get_reports(report_path):
    success = True
    all_details = {"failure": [], "skipped": [], "error": [], "warning": []}
    levels = ["unit", "integration"]
    summaries = {lvl: [] for lvl in levels}
    for suite in summaries.keys():
        for path in sorted(Path(report_path).glob(f"*_*_*_{suite}_report.xml")):
            print(path)
            summary, details = parse_junit(path)
            summaries[suite].append(summary)
            all_details["failure"].extend(details["failure"])
            all_details["skipped"].extend(details["skipped"])
            all_details["error"].extend(details["error"])
    success = all([s["outcome"] == "success" for sm in summaries.values() for s in sm])

    return summaries, all_details, success


def generate_summary(report_path, repo_name, package, verbose):
    env = Environment(
        loader=FileSystemLoader("./scripts/ci/"), autoescape=select_autoescape()
    )
    template = env.get_template("report_template.md")
    summaries, details, success = get_reports(report_path)
    status = "success" if success else "failure"
    return (
        template.render(
            repo_name=repo_name,
            summaries=summaries,
            details=details,
            show_skipped=False,
            package=package,
            status=status,
            verbose=verbose,
        ),
        success,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="Pytest Report generator",
        description="Generates markdown reports",
    )
    parser.add_argument("report_path")
    parser.add_argument("output_path")
    parser.add_argument("repo")
    parser.add_argument("package")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    summary, _ = generate_summary(args.report_path, args.repo, args.package, args.verbose)

    Path(args.output_path).write_text(summary)


if __name__ == "__main__":
    main()
