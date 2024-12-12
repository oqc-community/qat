#!/bin/bash

GITSTRING=$( grep -in -e "{\s*git\s*=" -e "{\s*path\s*=" -e "{\s*url\s*=" pyproject.toml )

if [[ -n $GITSTRING ]]; then
    echo "Git repo found in pyproject.toml. Please select a release version before merging."
    echo $GITSTRING
    exit 1
elif [[ -z $GITSTRING ]]; then
    echo "No depdendencies with git URL's found."
    exit 0
fi