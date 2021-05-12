# seedlingTools
This project separates images of multiple lettuce seedlings. 

This module currently only separates day/night images from an artificially lit environment and seed trays positioned horizontally. It does not currently separate seedlings within the trays, only the trays themselves. Future improvents might be made to allow faster image processing and neural network based image labeling. 

*Note: seedlingTools was specifically designed to work with my experimental setup.*

# Getting Started
You must have python installed. To get the requirements, run the following in the project directory.

```
pip install requirements.txt
```

Currently, the only file needed is core/seedlingTools.py. Options are available to run specific types of segmentation on images (day-night or tray).

```
# Viewing options
python core/seedlingTools.py --help
```

```
# Separate Day/Night Images
python core/seedlingTools.py --day "path/to/images"
```

```
# Separate seedling trays
python core/seedlingTools.py --tray "path/to/images"
```

```
# Plot a RGB histogram 
python core/seedlingTools.py --rgb "path/to/image/file"
```

```
# Plot a HSL histogram 
core/seedlingTools.py --hsl "path/to/image/file"
```

```
# Create a timelapse from file at 20 FPS
core/seedlingTools.py --timelapse "path/to/image/files"
```

# Features
## Day/Night Separation
This will separate day and images, placing them in newly created day and night directories respectively.

*Images will be moved from the original directory to the new labeled ones. This operation WILL overwrite image files with the same name so take care.*

## Tray Seperation
This will separate different seedling trays into folders labeled "1", "2", "3", "4", "5", and "6". It currently only supports images with 6 seedling trays. Your seed trays may not be the same size (in pixels) as mine in your images. You can tweak parameters in the tray_separator function to adjust this. 

*Images will be moved from the original directory to the new labeled ones. This operation WILL overwrite image files with the same name so take care.*

# Jupyter Notebooks
I used these notebooks for data exploration. Feel free to take a look, though it may be sloppyish.