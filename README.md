# IDEAL

This repository provides a reference implementation of *IDEAL* as described in the paper:<br>

> IDEAL: A Malicious Traffic Detection Framework with Explanation-Guided Learning<br>

## Installation and Execution

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

### Execution

Once the environment is configured, the programs can be run by the following command:

Train base models without explanation supervision

   ```sh
 python main.py --train_model --dataset cicids2017
   ```

Retrain base models with explanation supervision

   ```sh
 python main.py --train_with_exp_loss --dataset cicids2017 --exp_method inputgrad --exp_lambda 1.5
   ```

Test model

   ```sh
 python main.py --test_model --dataset cicids2017
   ```


