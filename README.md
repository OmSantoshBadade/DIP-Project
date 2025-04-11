# Brain Tumor Detection

A Python application for detecting brain tumors in MRI images using image processing and machine learning techniques.

## Features

- Image preprocessing for MRI scans
- Feature extraction from brain MRI images
- Machine learning model (SVM) for tumor detection
- User-friendly GUI for uploading and analyzing MRI images
- Real-time tumor detection results

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Pillow (PIL)
- Tkinter

## Installation

1. Clone this repository:
```
git clone https://github.com/OmSantoshBadade/DIP-Project.git
cd DIP-Project
```

2. Install the required packages:
```
pip install opencv-python numpy matplotlib scikit-learn pillow
```

## Usage

1. Prepare your dataset:
   - Create a `dataset` folder in the project directory
   - Inside the `dataset` folder, create two subfolders: `tumor` and `no_tumor`
   - Add brain MRI images to these folders (tumor images in `tumor` folder, non-tumor images in `no_tumor` folder)

2. Run the application:
```
python brain_tumor_detection.py
```

3. Use the GUI to upload and analyze new MRI images.

## Project Structure

- `brain_tumor_detection.py`: Main application file
- `dataset/`: Directory for training data
  - `tumor/`: Contains MRI images with tumors
  - `no_tumor/`: Contains MRI images without tumors

## License

This project is licensed under the MIT License - see the LICENSE file for details. 