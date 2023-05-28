#!/bin/sh

# Install pip if not already installed
sudo apt-get update
sudo apt-get install python-pip -y

# Install torchvision
pip install tabulate > /dev/null 2>&1
pip install pycocotools > /dev/null 2>&1
pip install torchvision > /dev/null 2>&1
pip install opencv-python > /dev/null 2>&1

# argument exception catching
if [ $# -eq 0 ]
then
    echo "Argument needed: name of your config file"
    exit 1
fi

# make saved_model directory if it doesn't exist
mkdir -p saved_models

# Only argument is config file
arg1=$1
echo "You are using: $arg1"

python main.py $arg1
