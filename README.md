# CMSC 265 Exercise 1 - Smooth transition from one image to another

## Requirements

1. Python 3.5.x
2. OpenCV 3.x
3. DLib 1.9.x
4. Boost Python (Required by DLib)

_Note: Make sure that Python is compiled with framework enabled when installing on macOS systems._

## Installing dependencies

This project requires a working installation of [OpenCV 3](http://opencv.org/). Please install this first before installing
the project dependencies.

Dependencies of this project can be installed via [PIP](https://pypi.python.org/pypi/pip).

Follow these steps to install the setup the application:

1. Run the command `pyvenv venv` to setup a Python 3 virtual environment.
2. Activate the virtual environment by running `source venv/bin/activate`
3. Install the project dependencies by running `pip install -r requirements.txt`
4. Locate your OpenCV Python bindings and type the command `echo <Python OpenCV bindings path> >> ./venv/lib/<python version>/site-packages/opencv3.pth`

    Where:

    1. `<Python CV bindings path>`: is the site-packages folder inside the OpenCV installation.
    Please make sure you select the appropriate version of bindings that matches the Python version declared in the requirements.

    2. `<python version>`: is the version under your `venv` virtual environment folder

5. Create `data` folder using the command `mkdir data`
6. Create a symlink/copy of `haarcascades` folder of OpenCV to `data/opencv-cascades/haarcascades`
7. Download shape predictor file from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
8. Place and extract the shape predictor file to `data/shape-predictor`

## Running the Program

This program can be run using the command `./run.py`.
The output image files will be under the `out` directory generated at the project root.

This program also includes a command line tool for advanced usage. Run `./cli.py --help` for more information

## Creating Video

You may create video out of the image sequences by using third party software like Adobe Photoshop, Virtualdub
and many more but you can create it directly from the command line if you have `ImageMagick` and `ffmpeg` installed on your path.

I have setup a simple python script that utilizes `ImageMagick` and `ffmpeg`. You may run the command `./cli.py create_video --help` for more information.

## License

MIT


