#!/bin/bash

abort()
{
    echo "*** FAILED ***" >&2
    exit 1
}

if [ "$#" -eq 0 ]; then
    echo "Wrong argument is provided. Usage:
            '-local' to build local environment
            '-test' to run linter, formatter and tests
            '-docker' to build and run docker image"

elif [ $1 = "-local" ]; then
    trap 'abort' 0
    set -e
    echo "Running format, linter and tests"
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r ./requirements.txt

    black sonix tests
    pylint --fail-under=9.9 sonix tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov sonix -v tests
elif [ $1 = "-test" ]; then
    trap 'abort' 0
    set -e
    
    echo "Running format, linter and tests"
    source .venv/bin/activate
    black sonix tests
    pylint --fail-under=9.9 sonix tests
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov --log-cli-level=INFO sonix -v tests
elif [ $1 = "-docker" ]; then
    echo "Building and running docker image"
    docker stop sonix-container
    docker rm sonix-container
    docker rmi sonix-image
    # build docker
    docker build --tag sonix-image --build-arg CACHEBUST=$(date +%s) . --file Dockerfile.test
elif [ $1 = "-deploy-package" ]; then
    echo "Running Sonix package setup"
    pip install twine
    pip install wheel
    python setup.py sdist bdist_wheel
    rm -rf .venv_test
    python3 -m venv .venv_test
    source .venv_test/bin/activate
    pip install ./dist/sonix-0.2-py3-none-any.whl
    pytest --ignore=tests/benchmark --cov-fail-under=95 --cov sonix -v tests
    # twine upload ./dist/*
else
  echo "Wrong argument is provided. Usage:
            '-local' to build local environment
            '-test' to run linter, formatter and tests
            '-docker' to build and run docker image"
fi

trap : 0
echo >&2 '*** DONE ***'