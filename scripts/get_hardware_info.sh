#!/bin/bash

export LANG=en_US.UTF-8

# Detect OS and its version
detect_os_and_version() {
    if [ "$(uname)" == "Darwin" ]; then
        OS="Mac"
        OS_VERSION=$(sw_vers -productVersion)
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        OS="Linux"
        # Attempt to find a suitable command to retrieve OS version
        if type -p lsb_release >/dev/null; then
            OS_VERSION=$(lsb_release -d | sed 's/Description:[ \t]*//')
        elif [ -f /etc/os-release ]; then
            OS_VERSION=$(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"')
        else
            OS_VERSION="Unknown"
        fi
    else
        OS="Unknown"
        OS_VERSION="Unknown"
    fi
    echo "OS: $OS_VERSION"
}

# Get CPU information
get_cpu_info() {
    if [ "$OS" == "Mac" ]; then
        echo "CPU Model: $(sysctl -n machdep.cpu.brand_string)"
    elif [ "$OS" == "Linux" ]; then
        echo "CPU Model: $(grep 'model name' /proc/cpuinfo | uniq | cut -d: -f2 | xargs)"
    fi
}

# Get memory information and specs (if available)
get_memory_info() {
    if [ "$OS" == "Mac" ]; then
        total_mem=$(sysctl -n hw.memsize)
        echo "Memory: $(echo "$total_mem / 1024 / 1024" | bc) MB"
        # Mac doesn't easily expose memory speed/type via command line
    elif [ "$OS" == "Linux" ]; then
        echo "Memory: $(free -m | grep Mem | awk '{print $2}') MB"
    fi
}

# Get Docker version (if Docker is installed)
get_docker_version() {
    if type -p docker >/dev/null; then
        echo "Docker: $(docker --version | cut -d ' ' -f3 | tr -d ',')"
    else
        echo "Docker is not installed."
    fi
}

# Main function to run the info functions
main() {
    detect_os_and_version
    get_cpu_info
    get_memory_info
    get_docker_version
}

# Execute the main function
main

