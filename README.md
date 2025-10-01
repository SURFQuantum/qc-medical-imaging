# QC Medical Imaging
This repository contains code for QC Medical Imaging.

## Building the Virtual Environment
To build a virtual environment for this project, follow these steps:

1. **Install Python**: Ensure you have Python 3.8 or later installed on your system.
2. **Create a virtual environment**: Run the following command to create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure
qc-medical-imaging/
├── README.md
├── requirements.txt
├── QCNN/               # Quantum NN work
├── CNN/                # classical convolutional NNs
├── docs/               # docs
├── Liza_Experiments/   # Liza's original experiments
└── .gitignore

# Dataset
Download train and valid files from PatchCamelyon
or vector size 48 (4x4x3) on confluence: https://confluence.ia.surf.nl/display/QUANTM/Dataset
