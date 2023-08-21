# ERA1_Session13
## Assignment
    1. Move Custom Object Detection Model to Lightning first and then to Spaces such that: 
    Train the model so that all of these are true:
    1. Class accuracy is more than 75% 
    2. No Obj accuracy of more than 95% 
    3. Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78) 
    4. Ideally trained till 40 epochs 
    • Add these training features: 
    1. Add multi-resolution training - the code shared trains only on one resolution 416 
    2. Add Implement Mosaic Augmentation only 75% of the times 
    3. Train on float16 
    4. GradCam must be implemented. 
    • Things that are allowed due to HW constraints: 
    1. Change of batch size 
    2. Change of resolution 
    3. Change of OCP parameters 
    • Once done: 
    1. Move the app to HuggingFace Spaces 
    2. Allow custom upload of images 
    3. Share some samples from the existing dataset 
    4. Show the GradCAM output for the image that the user uploads as well as for the samples. 
    5. Mention things like: 
        1. classes that your model support 
        2. link to the actual model 
## Solution - Notebook [s13_assignment](https://github.com/sdev2030/ERA1_Session13/blob/main/s13_assignment.ipynb)
In this we will use Custom architecture from the class, trained in torch lightning to achieve class and object accuracy of more than 80% by the 40th epoch. Implemented Mosaic Augmentation only 75% of the times, OneCycle Policy, train using float16 with batch size of 16 and image resolution of 416.

Following are the model parameter, class and object accuracies achieved in training the model for 40 epochs.
- Model Parameters - 61.6M
- Class Accuracy - 85.67%
- No Object Accuracy - 98.18%
- Object Accuracy - 77.64%
- MAP - 0.5065

## Solution - Notebook [multi_scale_s13_assignment](https://github.com/sdev2030/ERA1_Session13/blob/main/multi_scale_s13_assignment.ipynb)
In this we will use Custom architecture modified to handle multi-resolution inputs, trained in torch lightning to achieve class and object accuracy of more than 80% by the 40th epoch. Implemented Mosaic Augmentation only 75% of the times, OneCycle Policy, train using float16 with batch size of 16 and image resolution from 256 to 448 size inputs in multiples of 32 randomly selected during each batch (modifed code in Training Step in model.py file). 

Following are the model parameter, class and object accuracies achieved in training the model for 40 epochs.
- Model Parameters - 61.6M
- Class Accuracy - 82.63%
- No Object Accuracy - 97.87%
- Object Accuracy - 74.59%
- MAP - 0.25769

Graph showing learning rate used in one cycle LR policy.
![LR finder Graph](https://github.com/sdev2030/ERA1_Session13/blob/main/images/onecycle_lr.png)

Graphs from training showing loss for train and test datasets
![Training Graphs](https://github.com/sdev2030/ERA1_Session13/blob/main/images/training_graphs.png)

Also the local version of app.py file (for spaces app) can be found at [app.py](https://github.com/sdev2030/ERA1_Session13/blob/main/app_with_gradCAM.py.ipynb)
