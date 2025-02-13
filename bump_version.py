#!/usr/bin/env python3

import json
import sys
import tomlkit

PROJECT_PATH = 'pyproject.toml'
SWITCHER_PATH = 'docs/switcher.json'


def update_toml(new_version):
    with open(PROJECT_PATH, 'r') as f:
        proj = tomlkit.load(f)

    proj['project']['version'] = new_version

    with open(PROJECT_PATH, 'w') as f:
        tomlkit.dump(proj, f)


def update_switcher(new_version):
    with open(SWITCHER_PATH, 'r') as f:
        switcher_list = json.load(f)

    latest = {
        'version': new_version,
        'name': f'{new_version} (latest)',
        'url': 'https://nklsxn.github.io/mqr/latest',
    }
    bumped = switcher_list[0]
    bumped_vsn = bumped['version']
    bumped['name'] = bumped_vsn # remove "(latest)"
    bumped['url'] = f'https://nklsxn.github.io/mqr/versions/{bumped_vsn}'

    del switcher_list[0]
    switcher_list.insert(0, bumped)
    switcher_list.insert(0, latest)

    with open(SWITCHER_PATH, 'w') as f:
        json.dump(switcher_list, f, indent=4)


new_version = sys.argv[1]
print(f'Updating version -> {new_version}')
update_toml(new_version)
update_switcher(new_version)
