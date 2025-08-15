#!/usr/bin/env bash

# if no automation, source it manually
if ! command -v direnv; then
    source .envrc  &>/dev/null
fi

# get dir name
directory=${VIRTUAL_ENV}

# overwrite if one virtual env already exist
if [ -d "${directory}" ]; then
    echo "Delete the existing ${directory}."
    rm -rf "${directory}"
fi

# use the same python version if available
preferred_python="python3.11"
if ! command -v ${preferred_python}; then
    preferred_python="python"
fi

# create virtual environment
echo "Create venv at ${directory}."
${preferred_python} -m venv "${directory}"

# install packages
source "${directory}/bin/activate"
pip install --upgrade pip
pip install -e .[test]
