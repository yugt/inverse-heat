# Inverse Heat Equation

This project contains scripts to run experiments for solving the inverse problem in heat equation to reconstruct thermal conductivity from temperature measurements using different sensor configurations.

## Prerequisites

Ensure you have the following dependencies installed:
- Python 3.11+
- torch
- torchvision
- torchdiffeq
- numpy
- pandas
- plotly
- argparse
- moviepy


## Running the Experiment

To run the `test2D-typewriter.py` script, use the following command:
```python
python test/test2D-typewriter.py --experiment <experiment_index> --epochs <number_of_epochs> --sensor-type <sensor_type> --num-static-sensors <number_of_static_sensors>
```

### Arguments
- experiment: The index of the experiment in the training set (default: 0).
- epochs: The number of epochs for training (default: 500).
- sensor-type: The type of sensors to use, either "moving" (4 sensors) or "static".
- num-static-sensors: The number of static sensors to use (16 or 64), only applicable if sensor-type is "static" (default: 16).


### Example
To run an experiment with the default settings:
```python
python test/test2D-typewriter.py --experiment 0 --epochs 500 --sensor-type moving
```

To run an experiment with static sensors:
```python
python test/test2D-typewriter.py --experiment 1 --epochs 1000 --sensor-type static --num-static-sensors 64
```

## Understanding the output

The script will generate the following outputs:

- PNG frames for each epoch and forward simulation.
- CSV files containing the basis, gradient, loss, and error data in the `log/data/<config>` directory.
- Video files combining forward and inverse processes in the `log/video/<config>` directory.
- PDF images of the forward and epoch results in the `log/image/<config>` directory.

The outputs will be saved in the log and tmp directories.

## Notes
- Ensure the data directory contains the necessary datasets (MNIST).
- Adjust memory limits if necessary using the resource module in the script.

For more details, refer to the comments and documentation within the `src/inverse_heat2D.py` and `test/test2D-typewriter.py` script.