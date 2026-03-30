#!/bin/bash

# VIRTUAL_ENV holds the environment location.


# check if that is set, if it is not, quit with a message
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment activated. Please activate one before running this script."
    exit 1
fi

# For now, lets assume debug.
RELEASE_TYPE=debug

NAME_IN_CARGO=$(grep -Po "(?<=name ?= ?\")([^\"]+)" Cargo.toml -m 1)

# This symlinks the target library to the python directory
# And then puts a pth file in the venv.

# cpython-313-x86_64-linux-gnu
PYTHON_SOABI=$(python3 -c "import sysconfig; t=sysconfig.get_config_vars('SOABI')[0];print(t)")
PYTHONVERSION_WITHDOT=$(python3 -c "import platform; t=platform.python_version_tuple();print(f'{t[0]}.{t[1]}')")

PACKAGE_NAME=${NAME_IN_CARGO}
SITE_PACKAGES=${VIRTUAL_ENV}/lib/python${PYTHONVERSION_WITHDOT}/site-packages
echo ${SITE_PACKAGES}

PYTHON_SOURCE_CODE_LOCATION="./${PACKAGE_NAME}/"
PYTHON_EXTRA_PATH=$(realpath .)

echo ${PYTHON_EXTRA_PATH} > ${SITE_PACKAGES}/${PACKAGE_NAME}.pth

# Now symlink the shared object into the python source code location.

TARGETLIBPATH=$(realpath ./target/${RELEASE_TYPE}/lib${PACKAGE_NAME}.so)

ln -s ${TARGETLIBPATH}  ${PYTHON_SOURCE_CODE_LOCATION}/${PACKAGE_NAME}.${PYTHON_SOABI}.so
