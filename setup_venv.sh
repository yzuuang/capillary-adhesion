#!/usr/bin/env bash

# get dir name
if ! command -v direnv; then
    source .envrc  &>/dev/null
fi
directory=${VIRTUAL_ENV}

# overwrite if one virtual env already exist
if [ -d "${directory}" ]; then
    rm -rf "${directory}"
fi
python3.11 -m venv "${directory}"

# install packages
source "${directory}/bin/activate"
pip install --upgrade pip
pip install -e .[test]
