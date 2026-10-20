# Example activation script for the simulator environment. Copy to scripts/evalopt/activate-sim.sh
# (or point SIM_ACTIVATE at your own) and adjust the paths. Sourced by the launchers.
#
# Measured setup: Python 3.10, torch 2.4.1+cu121, SAPIEN 3.0.0b1, mplib 0.2.1, cuRobo 0.7.8,
# installed with the upstream scripts/_install.sh (see docs/evaluation.md).

source /path/to/sim-env/bin/activate          # or: conda activate RoboTwin

# Headless rendering: SAPIEN needs the NVIDIA Vulkan ICD and EGL vendor files of the installed driver.
# If the system does not provide them, write the two JSON files yourself (they point at the driver's
# libGLX_nvidia.so.0 / libEGL_nvidia.so.0) and export:
# export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
# export __EGL_VENDOR_LIBRARY_FILENAMES=/path/to/10_nvidia.json

# export CUDA_HOME=/usr/local/cuda             # the CUDA toolkit cuRobo was built with
export PYTHONPATH="$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
