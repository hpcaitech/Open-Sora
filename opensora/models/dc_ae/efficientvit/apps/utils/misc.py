# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union

import yaml

__all__ = [
    "parse_with_yaml",
    "parse_unknown_args",
    "partial_update_config",
    "resolve_and_load_config",
    "load_config",
    "dump_config",
]


def parse_with_yaml(config_str: str) -> Union[str, dict]:
    try:
        # add space manually for dict
        if "{" in config_str and "}" in config_str and ":" in config_str:
            out_str = config_str.replace(":", ": ")
        else:
            out_str = config_str
        return yaml.safe_load(out_str)
    except ValueError:
        # return raw string if parsing fails
        return config_str


def parse_unknown_args(unknown: list) -> dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key, val = unknown[index], unknown[index + 1]
        index += 2
        if not key.startswith("--"):
            continue
        key = key[2:]

        # try parsing with either dot notation or full yaml notation
        # Note that the vanilla case "--key value" will be parsed the same
        if "." in key:
            # key == a.b.c, val == val --> parsed_dict[a][b][c] = val
            keys = key.split(".")
            dict_to_update = parsed_dict
            for key in keys[:-1]:
                if not (key in dict_to_update and isinstance(dict_to_update[key], dict)):
                    dict_to_update[key] = {}
                dict_to_update = dict_to_update[key]
            dict_to_update[keys[-1]] = parse_with_yaml(val)  # so we can parse lists, bools, etc...
        else:
            parsed_dict[key] = parse_with_yaml(val)
    return parsed_dict


def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        if key in config and isinstance(partial_config[key], dict) and isinstance(config[key], dict):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]
    return config


def resolve_and_load_config(path: str, config_name="config.yaml") -> dict:
    path = os.path.realpath(os.path.expanduser(path))
    if os.path.isdir(path):
        config_path = os.path.join(path, config_name)
    else:
        config_path = path
    if os.path.isfile(config_path):
        pass
    else:
        raise Exception(f"Cannot find a valid config at {path}")
    config = load_config(config_path)
    return config


class SafeLoaderWithTuple(yaml.SafeLoader):
    """A yaml safe loader with python tuple loading capabilities."""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderWithTuple.construct_python_tuple)


def load_config(filename: str) -> dict:
    """Load a yaml file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    return yaml.load(open(filename), Loader=SafeLoaderWithTuple)


def dump_config(config: dict, filename: str) -> None:
    """Dump a config file"""
    filename = os.path.realpath(os.path.expanduser(filename))
    yaml.dump(config, open(filename, "w"), sort_keys=False)
