# AIDx Console

AIDx Console is an interpretability testing framework designed to run with Python 3.8 or higher.

## Prerequisites

Make sure you have Python 3.8 or higher installed on your system. You can check your Python version by running:

```bash
python --version
```

## Installation
To install the required packages, navigate to the project directory and run:

```python
pip install -r requirements.txt
```

## Configuration
Before running the application, perform the following setup:

1. Set the Import Path: Update the path in 'interpretability_testing/interpretability/interpretability_test.py' with the path to your project's directory.

```python
sys.path.append("/<YOUR_PATH>/project/Interpretability_testing/")
# Replace <YOUR_PATH>/project/ with the actual path to where the Interpretability_testing directory resides on your system.
```

2. Python Command in app.py: Set the path to your Python executable in the app.py file according to your system's configuration. Replace <YOUR_PYTHON> with the appropriate command (python3 or python).
```python
# e.g.
program_b_args = ['<YOUR_PYTHON>', '../../aisg_demo/backend/api.py', '-c', caseId, '-m', requestData['modelFile'].split("\\")[-1], '-d', requestData['datasetFile'].split("\\")[-1], '-k', clusterSize, '-t', maxTokens]
# fill <YOUR_PYTHON> as e.g., python3, python
```

## Launch the Application
```python
python sav_demo/main/app.py
```

## Citation
```bash
@article{zhang2022neural,
  title={Which neural network makes more explainable decisions? An approach towards measuring explainability},
  author={Zhang, Mengdi and Sun, Jun and Wang, Jingyi},
  journal={Automated Software Engineering},
  volume={29},
  number={2},
  pages={39},
  year={2022},
  publisher={Springer}
}
```
