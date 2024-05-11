# Deep Learning Final Completion Spring 2024
### Author: Yibin Wang, Jiaxi Xie, Mrunal Sarvaiya
### Group 11

## Setup Instructions

1. **Download Dataset**:
   - Please download the final completion dataset first.

2. **Organize Dataset**:
   - Place the hidden dataset into the dataset folder. Structure the dataset folder as follows:
     ```
     |-- dataset_student
         |-- train
         |-- val
         |-- unlabeled
         |-- hidden
     ```

3. **Relevant Files**:
   - The project involves several key files:
     - `video_dataset.py`: Handles video dataset operations.
     - `rrunet.py`: Contains the U-Net based segmentation model.
     - `simvp.py`: Includes the implementation of the modified SimVP.

## Training Pipeline

Follow these steps to train and test the models:

1. **Train U-Net for Pseudo Labeling**:
   - Use `rrunet_seg.ipynb` to train the U-Net based segmentation model for pseudo labeling.

2. **Label Unlabeled and Hidden Data**:
   - Load the trained U-Net model or provided checkpoints in `unlabel_data_labeling.ipynb` to generate masks for the unlabeled and hidden datasets.

3. **Train Modified SimVP**:
   - Train the modified SimVP for mask prediction using `simvp_frame_pred.ipynb`.

4. **Finetune SimVP**:
   - Optionally, finetune the trained SimVP for additional epochs using `simvp_finetune.ipynb`.

5. **Test and Output Results**:
   - Test the trained SimVP model and output the results (`.pt` file) for the hidden dataset using `future_frame_seg.ipynb`.

## Support

If you have any questions regarding the code or its execution, please contact via [email](mailto:yw4145@nyu.edu).

