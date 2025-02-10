# Clinically-Inspired Hierarchical Multi-Label Classification of Chest X-rays with a Penalty-Based Loss Function

This project implements an **efficient**, **single-model hierarchical** classifier for chest X-ray (CXR) image analysis, grounded in clinical insights. By leveraging deep learning techniques, the model predicts multiple pathology labels with high accuracy, while offering visual explanations and uncertainty estimation for each prediction. 
This clinically-informed hierarchical approach enhances interpretability and aligns with diagnostic workflows, addressing limitations in traditional classification models.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

#### Key Features
  - **Dataset**: Trained on the **CheXpert** dataset, utilizing **VisualCheXbert** labels for improved pathology localization. 
  - **Custom Loss Function**: Implements a **penalty-based hierarchical binary cross-entropy** loss to enforce clinically relevant label relationships.
  - **Model Architecture**: Developed on **DenseNet121**, a deep convolutional neural network known for its efficient feature propagation and superior performance in medical imaging tasks. 
  - **Uncertainty Quantification**: Incorporates **Monte-Carlo** uncertainty calculations to provide uncertainty estimates for model predictions.
  - **Explainability**: Supports **Class Activation Map (CAM)** heatmaps using **Grad-CAM** method for visualizing model attention, enhancing interpretability for clinical applications. 
  - **Multi-view Support**: Predicts pathologies from both **frontal and lateral** view CXRs. 
  - **API Integration**: The model is deployed using an async **FastAPI** server, providing a lightweight, scalable inference service with **JSON** input/output.

## Table of Contents
1. [Quickstart](#quickstart)
2. [Project Structure](#project-structure)
3. [Installation and Usage](#installation-and-usage)
4. [API](#api)
5. [Training the Model](#training-the-model)
6. [Configurations](#configurations)
7. [Dataset](#dataset)
8. [Citation](#citation)
9. [Demo](#demo)
10. [Contributing](#contributing)
11. [Acknowledgements](#acknowledgements)
12. [License](#license)

## Quickstart
Run a basic inference on CXR images in just a few steps:

1. **Clone** the repository and install dependencies:
   ```bash
   git clone https://github.com/the-mercury/CIHMLC.git
   
   cd CIHMLC
   
   pip install -r requirements.txt
   ```
2. **Start** the server and make a prediction:
    ```bash
    uvicorn src.cxr_inference_app:app --host 0.0.0.0 --port 8000
     
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"cxr_base64": "<base64-encoded-image>"}'
    ```

_For detailed setup instructions and Docker deployment, see [Installation and Usage](#installation-and-usage)._

## Demo
Here’s an example of a CAM heatmap generated by the model:

![Sample Heatmap](https://raw.githubusercontent.com/the-mercury/CIHMLC/main/default_model_files/heatmaps/CAM_ch_0.5_cc_0.5_cmap_hot_r_da_False/Abnormal_cam.png)

_(All the CAM heatmaps generated by the default model for this image can be found [here](https://github.com/the-mercury/CIHMLC/tree/main/default_model_files/heatmaps/CAM_ch_0.5_cc_0.5_cmap_hot_r_da_False))_

**Ground Truth (GT) Pathologies:**
- Atelectasis
- Cardiomegaly
- Edema
- Enlarged Cardiomediastinum
- Lung Opacity
- Pleural Effusion

For a detailed walkthrough of the inference process, refer to the [API](#api) section.

## Project Structure
    .
    ├── data/                                               # Directory for CheXpert dataset and label files
    │   ├── CheXpert/                                       # Patient images
    │   ├── train_labels.csv                                # Training labels
    │   └── val_labels.csv                                  # Validation labels
    │
    ├── docker/                                             # Docker configuration files
    │   ├── Dockerfile
    │   └── docker-compose.yml
    │
    ├── fresh_models/                                       # Trained model checkpoints
    │   └── model_name.keras
    │
    ├── logs/                                               # Logs, heatmaps, and training metrics
    │   ├── heatmaps/                                       # Generated CAM visualizations
    │   └── tensorboard/                                    # TensorBoard logs for visualization
    │
    ├── src/                                                # Source code
    │   ├── helpers/                                        # Utility scripts
    │   │   └── cam.py                                      # Visualization
    │   ├── data/                                           # Data loaders
    │   │   └── chexpert_data_loader.py
    │   ├── config.py                                       # Configuration file
    │   ├── cxr_inference_app.py                            # FastAPI application for model inference
    │   ├── hierarchical_binary_cross_entropy.py            # Custom loss function
    │   └── train.py                                        # Script to train the model
    │
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt

*(For a complete structure, refer to the repository)*

## Installation and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- Docker
- Docker Compose

### Inference 

- #### Running with Docker
  1. Clone the repository:
        ```bash
        git clone https://github.com/the-mercury/CIHMLC.git
        cd CIHMLC
        ```
  2. Build and run the Docker containers:
        ```bash
        cd docker
        docker compose up --build   # Make sure Docker and Docker Compose are installed
        ```

- #### Running without Docker (Custom Configuration)
  1. Clone the repository:
      ```bash
      git clone https://github.com/the-mercury/CIHMLC.git
      cd CIHMLC
      ```
  2. Install the requirements:
     ```bash
      pip install -r requirements.txt
      ```
  3. To start the FastAPI prediction service:
     ```bash
     uvicorn src.cxr_inference_app:app --host [IP] --port [port_num] --workers [num_workers]
     ```
  4. Make a prediction:
     ```bash
      curl -X POST "http://[IP]:[port_num]/predict" -H "Content-Type: application/json" -d '{"cxr_base64": "<base64-encoded-image>"}'
     ```

_Note: Replace <base64-encoded-image> with actual base64 data. The `cxr_base64` field should contain the Base64-encoded string of the CXR image. You can convert an image using Python’s `base64` library or online converters._ 
    
```python
import base64

with open("path_to_cxr_image.jpg", "rb") as img_file:
  base64_string = base64.b64encode(img_file.read()).decode('utf-8')

print(base64_string)  # Use this string in the API request
```

## API
The API exposes a `/predict` endpoint to make predictions on CXR images. The request format is as follows:

### Request Format
```
curl -X POST "http://[IP]:[port]/predict" -H "Content-Type: application/json" -d '{"cxr_base64": "<base64-encoded-image>"}'
```
_Replace [IP]:[port] with your designated number._

```JSON
   {
       "cxr_base64": "<base64-encoded-image>"
   }
```

### Response Format

```JSON
   {
  "success": true,
  "heatmap": {
    "Atelectasis": "<base64-encoded-heatmap>",
    "Cardiomegaly": "<base64-encoded-heatmap>",
    ...
  },
  "prediction_mean": {
    "Atelectasis": 0.5,
    "Cardiomegaly": 0.8,
    ...
  },
  "prediction_variance": {
    "Atelectasis": 0.03,
    "Cardiomegaly": 0.05,
    ...
  },
  "inference_duration": 20
}
```
#### * The CAM heatmaps are also stored in `/logs/heatmaps/[model_name]` directory in `.png` format.

### Training the model
To train the model, execute the following steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/the-mercury/CIHMLC.git
    cd CIHMLC
    ```
2. Install the requirements:
   ```bash
    pip install -r requirements.txt
    ```
3. Start the training:
   ```bash
   python src/train.py
   ```


**NOTE:** 
- The model will be trained based on configurations specified in `config.py`, and the new models will be stored in `fresh_models/[model_name]` directory including the **best AUROC**, and the **best loss** checkpoints. 
- If you need to replace the default model, you can move the newly trained model to the `src/assets/models` directory and `rename` it, or **update** the `model directory and name` settings in `config.py`. 
- To monitor training performance, logs will be saved in the `/logs/tensorboard` directory, which can be visualized using `TensorBoard`:
   ```bash
   tensorboard --logdir=logs/tensorboard
   ```

## Configurations
The configuration is managed through the `Config` class in `src/config.py`. Key parameters include:
- Device settings
- Project-specific settings
- Model architecture and training settings
- Data paths and preprocessing options

## Dataset
This project used the **CheXpert** dataset, and the **VisualCheXbert** labels. For more details on these resources, please refer to the following publications:

*Note: The CheXpert dataset requires registration and approval from the authors. Follow [this link](https://stanfordmlgroup.github.io/competitions/chexpert/) for access.*

```bibtex
    @inproceedings{irvin2019chexpert,
      title={CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
      author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={33},
      pages={590--597},
      year={2019}
    }

    @inproceedings{smit2022visualchexbert,
      title={VisualCheXbert: Adaptation of CheXbert for Improved Performance in Localizing Pathologies in Chest X-rays},
      author={Smit, Alice and Taylor, Aaron and Srinivasan, Bharath and Bindal, Akshay and Trivedi, Hiren and Ma, Maxwell and Ng, Andrew Y and Piech, Chris and Rajpurkar, Pranav},
      booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      year={2022},
      pages={12345-12356}
    }
```

## Citation
**If you find this work useful, please cite:**

For more details, see the full paper on [arXiv](https://doi.org/10.48550/arXiv.2502.03591).

```bibtex
    @article{asadi2025cihmlc,
        title={Clinically-Inspired Hierarchical Multi-Label Classification of Chest X-rays with a Penalty-Based Loss Function},
        author={Asadi, Mehrdad and Sodoké, Komi and Gerard, Ian J. and Kersten-Oertel, Marta},
        journal={arXiv preprint arXiv:2502.03591},
        year={2025},
        pages={1--9},
        doi={10.48550/arXiv.2502.03591},
        url={https://doi.org/10.48550/arXiv.2502.03591},
    }
```

## Contributing
_**Contributions are welcomed!**_

**To get involved, please follow these steps**:
1. **Fork** the repository.
2. Create a **new branch**: 
   ```bash 
   git checkout -b my-feature-branch
   ```
3. **Commit** your changes:
   ```bash 
   git commit -am 'Add new feature'
   ```
4. **Push** to the branch: 
   ```bash 
   git push origin my-feature-branch
   ```
5. Submit a **pull request** for review.

## Acknowledgements
Special thanks to the **Stanford ML Group** for the CheXpert dataset and to the creators of VisualCheXbert.

## License
This project is licensed under the `MIT License`. See the `LICENSE` file for more details.