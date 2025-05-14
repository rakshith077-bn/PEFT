#!/bin/bash

# Display the heading
echo "Installing required dependencies..."
echo

pip install --upgrade pip

pip install -r requirments.txt
if [ $? -ne 0 ]; then
  echo "There was some error while installing dependencies. Exiting"
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Using macOS environment. Recommended (tensorFlow-macOS + tensorFlow-metal) for GPU training"
    echo
fi

echo
echo "Dependencies installed successfully."

echo
echo
python -c "from src.header import welcome_note; welcome_note()"
python -c "from src.header import main_intro; main_intro()"
