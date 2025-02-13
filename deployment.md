mqrpy deployment guide
----------------------

1.  Make changes<br>
    Use commit messages with the following prefixes:<br>
    * `enh #<issue num>: <summary of enhacement>`
    * `bug #<issue num>: <summary of bug>`
    * `doc #<issue num>: <summary of documentation change>`
    * `mnt #<issue num>: <summary of maintenance action>`
    where __issue_num__ is a github issue number.

2.  At the next deployment, update the versions in `pyproject.toml` and `docs/switcher.json`
    using `bump_version.py <new_version>`.

    Commit the version changes with the message:<br>
    `version -> <new_version>`.

    Tag the version using:<br>
    `git tag -a <new_version> <new_version>`.

3.  Push the version update and tag, which triggers the github pages workflow to build new documentation:<br>
    `git push && git push --tags`.

4.  Build the and upload packages:<br>
    `python3 -m build`<br>
    `python3 -m twine upload --repository pypi dist/mqrpy-<new_version>-py3-none-any.whl dist/mqrpy-<new_version>.tar.gz`
