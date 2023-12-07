# Tag Identification

## Project Description

This project aims to develop a YOLOv8 model to recognize employees based on their employee tags within video footage. To achieve this, the project utilizes a combination of video processing, image annotation, and transfer learning techniques.

## Project Methodology

1. **Frame Extraction**: 
    
    The video is first processed to extract individual frames, capturing each moment in time.

2. **Image Download and Annotation**: 
    
    These frames are then downloaded and meticulously annotated with relevant information for training purposes. This creates a labeled dataset for the machine learning model to understand.

3. **Enhanced Training Dataset**: 
    
    In Roboflow, the annotated images are further enriched through augmentation techniques. This process expands the dataset by generating variations of existing images, improving the model's generalizability and robustness.

4. **Data Acquisition and Model Training**: 
    
    Finally, the complete dataset is accessed via an API and used to train the YOLOv8 model. 

5. **Deployment and Application**:

    The models are leveraged into an application to detect employees with the employee tag.

## Appication Flow:

1. Video Input:

    The video is 

2. Model Inference:

    Each frame is passed through the People detection model and Tag detection model to detect the people and tags located in the frame respectively.

3. Object Detection:

    When an employee tag is detected, the closest person to the tag will be located and its relevant information is taken.

4. Object Tracking:

    The tracking id of the person last detected with a tag will be noted until a new tag is detected.

5. Output:

    A video containing the annotated bounding box of the employee is created. The frames and coordinates of the employee is outputted in the terminal.

## Installation & Usage

Files
'''
https://github.com/NigelGoh16/Tag_Identification/tree/master
'''

Dependencies installation:
'''
pip install -r requirements.txt
'''

To run the script to execute the employee tag detection:
'''
python main.py
'''

### Custom Data

For custom models:
    
    Any models can be retrained with your own custom data in the Tag_Identification.ipynb notebook.

For custom data:

    New data can be processed through the python main.py
    (Note: Update the video paths before processing any new data)

## Output Example

Frame and Coordinates Output:
![](https://drive.google.com/uc?export=view&id=16MJ8jLmZZRnRf2xYQQoB7lJZ6kyG3E1e)
![](https://drive.google.com/uc?export=view&id=1fasRg9YScCreGU6TzIEwagLmCT-Wnp6K)

Video Output:
![](https://drive.google.com/uc?export=view&id=1fEPtGWw-LljcSgwWDkmBKq8z-5KSfdlN)

## Future Work

There are several known limitations to this approach in which are:

1. The tracking of the employee is refreshed everytime a new tag is detected.
This is alright with 1 employee however when considering multiple employees, whenever a new employee tag is detected and the prior employee being tracked with his tag no longer visible will not be tracked anymore.

2. The tracking id of the employee is reassigned 

