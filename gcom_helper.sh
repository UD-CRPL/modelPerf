#!/bin/bash
# GCoM Automation Script
#
# This script automates cloning and building the GCoM project.
# Usage: ./gcom_helper.sh [git|build]
#   git   - Clone the GCoM repository into the current directory (creates a 'gcom' folder).
#   build - Build gzstream, Boost Serialization, and compile the GCoM project.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 {git|build}"
    exit 1
fi

if [ "$1" == "git" ]; then
    echo "Cloning the GCoM repository from GitHub..."
    git clone https://github.com/yonsei-hpcp/gcom.git
    if [ $? -eq 0 ]; then
        echo "GCoM repository successfully cloned into $(pwd)/gcom/"
    else
        echo "Error: Failed to clone GCoM repository."
        exit 1
    fi

elif [ "$1" == "build" ]; then
    if [ ! -d "gcom" ]; then
        echo "Error: GCoM source directory not found. Please run '$0 git' first to clone the repository."
        exit 1
    fi

    cd gcom

    echo "Building gzstream..."
    ( cd third-party/gzstream && make )

    echo "Building Boost Serialization..."
    ( cd third-party/boost_1_86_0 && ./bootstrap.sh && ./b2 --with-serialization )

    echo "Building the GCoM project (this may take a few minutes)..."
    make -j

    if [ -f "bin/GCoM" ]; then
        echo "Build completed successfully. The GCoM binary is located at $(pwd)/bin/GCoM"
    else
        echo "Build may have encountered errors. Please check the output above for details."
        exit 1
    fi

else
    echo "Usage: $0 {git|build}"
    exit 1
fi
