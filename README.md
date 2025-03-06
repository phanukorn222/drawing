# Hand Drawing with Mediapipe

This project uses **Mediapipe** from Google and **OpenCV** to create an application that allows drawing with hand gestures. It detects hand landmarks through a webcam and enables saving or clearing the drawing by specific hand gestures.

## Features

- Draw on the board using the index finger.
- Stop drawing when the index and middle fingers are raised.
- Save the drawing by clicking in the "Save" zone.
- Clear the drawing board by clicking in the "Clear" zone.

## Installation

This project requires **Python** and some libraries that can be installed using the following command:

### Install required libraries

```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script to start the application:

   ```bash
   python main.py
   ```

2. Use your webcam to draw on the screen with your hand gestures.

## Training Data

The training data for different shapes is stored in the `data/` directory. Each `.npy` file corresponds to a different shape.

## Model

The trained model is saved as `quickdraw_model.pth`.

## Development

The source code is located in the `src/` directory. It includes the following files:

- `config.py`: Configuration settings for the project.
- `model.py`: Model definition.
- `utils.py`: Utility functions.

## Virtual Environment

Create the virtual environment

```bash
python -m venv env
```

and Activate

```bash
env\Scripts\activate
```
