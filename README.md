# Cancer Diagnosis in Histopathological Images: A CNN-based Approach (Paper Reproduction)

This project is a PyTorch-based implementation and reproduction of a Convolutional Neural Network (CNN) model for cancer diagnosis in histopathological images, as described in the paper "Cancer diagnosis in histopathological image: CNN based approach". The model is designed to classify breast cancer biopsy images from the BreakHis dataset into benign and malignant tumors.

## Dataset

This project utilizes the **BreakHis dataset**. The dataset consists of 7,909 microscopic images of breast tumor tissue, collected from 82 patients. The images are categorized into two main classes: Benign and Malignant.

The `main.ipynb` notebook expects the dataset to be organized in a directory named `datasets/` in the root of the project.

## Model Architecture

The CNN architecture is defined in `CNN.py` and is designed to follow the specifications of the original paper.

**Feature Extraction Layers (Convolutional):**

| Layer attribute  | L1   | L2   | L3   | L4   | L5   | L6   |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| **Type** | conv | pool | conv | pool | conv | pool |
| **Channel** | 32   | –    | 64   | –    | 128  | –    |
| **Filter Size** | 5×5  | –    | 5×5  | –    | 5×5  | –    |
| **Conv. stride** | 1×1  | –    | 1×1  | –    | 1×1  | –    |
| **Pooling size** | –    | 3×3  | –    | 3×3  | –    | 3×3  |
| **Pooling stride**| –    | 1×1  | –    | 1×1  | –    | 1×1  |
| **Padding size** | same | none | –    | none | –    | none |
| **Activation** | ReLU | –    | ReLU | –    | ReLU | –    |

**Classifier Layers (Fully Connected):**

| Layer Attribute | FC-1 | FC-2 | FC-3    |
| --------------- | ---- | ---- | -------|
| **No of nodes** | 64   | 64   | 2      |
| **Activation** | ReLU | ReLU | Softmax|

The implementation uses PyTorch's `nn.Module` and includes `Conv2d`, `MaxPool2d`, `Linear`, and `BatchNorm1d` layers.

## Requirements

The project is built using Python and requires the following libraries:

  * `torch`
  * `torchvision`
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `Pillow (PIL)`
  * `tqdm`
  * `jupyter`

## Usage

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (Note: You may need to create a `requirements.txt` file based on the list above.)

3.  **Download and prepare the dataset:**
    Download the BreakHis dataset and place the image folders into a `datasets/` directory in the project's root.

4.  **Run the notebook:**
    Launch Jupyter Notebook and run the cells in `main.ipynb` to load the data, train the model, and evaluate its performance.

## Project Structure

  * `CNN.py`: Contains the definition of the `BreakHisCNN` model class.
  * `main.ipynb`: A Jupyter Notebook that covers the entire machine learning workflow, including:
      * Data loading and exploration.
      * Data preprocessing and augmentation.
      * Model training and validation.
      * Performance evaluation and visualization of results.

## Results

The notebook provides a comprehensive evaluation of the model's performance, including:

  * **Training and validation curves** for loss and accuracy.
  * **A classification report** with precision, recall, and F1-score for each class.
  * **A confusion matrix** to visualize the model's predictions.
  * **Final test accuracy**.

The training process incorporates techniques like weighted sampling to handle class imbalance and gradient scaling for mixed-precision training.

## Reference

This project is a reproduction of the following paper:

  * **Title:** Cancer diagnosis in histopathological image: CNN based approach
  * **Authors:** Sumaiya Dabeer, Maha Mohammed Khan, Saiful Islam
  * **Publication Date:** 10 Oct 2019
  * **Link:** [https://www.sciencedirect.com/science/article/pii/S2352914819301133](https://www.sciencedirect.com/science/article/pii/S2352914819301133)