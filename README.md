
## About

Celiac Detection IITD-AIIMS is a project focused on the detection of celiac disease using AI technologies. This project aims to provide an accurate and efficient method for diagnosing celiac disease, contributing to the advancement of medical diagnostics and helping practitioners calibrate and use the images in an end-to-end fashion

## Features

- Feature 1: Accurate celiac disease prediction
- Feature 2: Image Calibration module
- Feature 3: Image Annotation Measurement 
- Feature 4: Efficient AI-powered analysis
- Feature 5: User-friendly interface


## Getting Started

Follow these instructions to set up and use the Celiac Detection IITD-AIIMS project on your local machine.

### Prerequisites

- Ubuntu / Windows
- conda
- Python 3.9


### Installation

1. Clone the repository and move to the project directory:
   ```bash
   git clone https://github.com/hvarS/medical-app.git
   cd medical-app
2. Create a Conda environment named 'anti-celiac':
   ```bash
   conda create -n anti-celiac python=3.9
   conda activate anti-celiac
3. Install the dependencies from 'requirements.txt':
    ```bash
   pip install -r requirements.txt
### Usage 

#### Note 
Before usage, please download the model and add the weights to the project directory 
Use:
`mkdir weights`


Follow the below steps to run the app on your local machine:

1. Activate the 'anti-celiac' environment:
    ```bash
    conda activate anti-celiac
2. Run the app:
    ```bash
    python run_app.py

### Contributing

We welcome contributions from the community to enhance the Celiac Detection IITD-AIIMS project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m "Add some feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Create a pull request describing your changes.


### Error Handling
Change:
    ```bash
    return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
    recompute_scale_factor=self.recompute_scale_factor)
To:
    ```bash
  return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)