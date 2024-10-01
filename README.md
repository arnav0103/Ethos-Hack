# Ethos-Hack 

## Overview
This project utilizes the GFPGAN model for enhancing and restoring images, particularly focusing on face restoration. It extracts frames from videos, processes images, and saves the results in specified directories.

## Credits
This project utilizes the GFPGAN model created by the [GFPGAN team](https://github.com/TencentARC/GFPGAN).

## Requirements
You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Download the GFPGAN Model**:
   The `run.py` script will automatically download the GFPGAN model if it is not already present in the directory.

2. **Prepare Input Data**:
   Place your video files in a folder named `video`. The program will extract frames from these videos.

3. **Run the Program**:
   Execute the following command to start the processing:
   ```bash
   python run.py
   ```

4. **Output**:
   - The processed images will be saved in the `results` directory.
   - Cropped and restored faces will be saved in `results/cropped_faces` and `results/restored_faces`, respectively.
   - Comparison images will be saved in `results/cmp`.
   - The restored images will be saved in `results/restored_imgs`.

5. **Examples**:
   - **Example 1**: Contains a video file `example1/cctv1.mp4`.  After running the program, the results will be saved in the `example1/results` folder.
   - **Example 2**: Similar to Example 1, it also contains a video file `example2/cctv2.mp4` and its results folder.

## Customization
You can customize the input and output directories by modifying the arguments in the `inference_gfpgan.py` script. The default input directory is `inputs/whole_imgs` and the output directory is `results`.

