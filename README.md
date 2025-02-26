# IDEAL

This repository provides a reference implementation of *IDEAL* as described in the paper:<br>

> IDEAL: A Malicious Traffic Detection Framework with Explanation-Guided Learning<br>

## Installation and Execution

### From Source

Start by grabbing this source code:

```
git clone https://github.com/Eternal-548/IDEAL.git
```

### Environment

It is recommended to run this code inside a `conda` environment with `python3.10`.

- Create environment:

  ```sh
  conda create -n IDEAL python=3.10
  ```

- Activate environment:

  ```sh
  conda activate IDEAL
  ```

### Requirements

Latest tested combination of the following packages for Python 3 are required:

- pytorch (2.0.0)
- captum (0.7.0)
- numpy (1.24.3)

To install all the requirements, run the following command:

```
python -m pip install -r requirements.txt
```

### Execution

Once the environment is configured, the programs can be run by the following command:

   ```sh
 python Main.py
   ```

### Citation

If you use this code for your research, please cite our paper.

*Thank you for your interest in our research.*
