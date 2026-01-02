# 🔥 Fire Detection System using YOLO 🛑


This project implements a **fire and smoke detection system** using the **YOLO (You Only Look Once)** object detection model. The system is built with the **Ultralytics YOLO library** and is designed to detect **smoke and fire** in images and videos.  

The notebook (`fire-detection-using-yolo-model.ipynb`) demonstrates the complete pipeline from **data preprocessing** to **model deployment**.  

<img width="342" height="627" alt="image" src="https://github.com/user-attachments/assets/f7022d66-de23-44a0-8083-048c1d6535f7" />
<img width="351" height="629" alt="image" src="https://github.com/user-attachments/assets/2c93f35d-2402-47f5-bd0f-ac58e0fabc25" />

## 🚀 Features

- 🧹 **Data Cleaning**: Automatically identifies and removes corrupted or problematic images.  
- 🖥️ **YOLO Model Training**: Trains a custom YOLO model for smoke and fire detection.  
- 📊 **Model Validation**: Evaluates the trained model on validation and test sets.  
- 🖼️ **Visualization**: Displays sample images with bounding boxes for detected objects.  
- 💾 **Export**: Saves trained models in multiple formats (PyTorch, ONNX, etc.).  



## 🛠️ Prerequisites

Before running this project, ensure you have:  

- 🐍 Python 3.8+  
- 📓 Jupyter Notebook or JupyterLab  
- ⚡ A CUDA-compatible GPU (recommended for faster training)  
- 💽 Sufficient disk space for datasets and models  



## 📦 Installation

1. **Clone or Download the Repository**  
   - Ensure the project files are in your workspace, specifically in the `Fire Detection system` folder.  

2. **Install Required Packages**  
   - Run:  
     ```bash
     !pip install ultralytics
     ```  
   - Other dependencies (OpenCV, NumPy, Matplotlib, etc.) are usually pre-installed.  

3. **Set Up Environment**  
   - Activate your virtual environment if using one.  
   - For GPU support, ensure **PyTorch with CUDA** is installed (Ultralytics handles this).  



## 📁 Dataset

The project uses the **"Smoke Fire Detection YOLO"** dataset for training object detection models.  


### 🧹 Data Cleaning
- Removes bad images listed in `bad_list`.  
- Detects corrupted images with OpenCV.  
- Copies valid images and labels to a clean directory.  
- Generates a **report CSV** of actions taken.  


## ⚡ Usage

### Running the Notebook
1. Open `fire-detection-using-yolo-model.ipynb` in Jupyter.  
2. Execute cells in order:  
   - 📥 Import libraries  
   - 🛣️ Define paths  
   - 🗑️ List bad images  
   - 📂 Create directories & clean data  
   - 📑 Generate data report  
   - 📝 Create `data.yaml`  
   - 🖼️ Visualize sample images  
   - 📦 Install Ultralytics  
   - 🔄 Import YOLO  
   - 🚀 Load pre-trained model  
   - 🏋️ Train the model  
   - ✅ Validate the model  

3. Adjust training parameters in `model.train()` (e.g., epochs, image size).  
4. Modify class names in `data.yaml` if needed (`smoke` and `fire`).  



### ✅ Validation and Testing
- Evaluate metrics: **mAP, precision, recall**.  
- Test on new data by updating paths.  


## ⚠️ Troubleshooting
- 🛣️ **Path Issues**: Update `/kaggle/input/` paths for local setup.  
- 💾 **Memory Errors**: Reduce batch size or image size.  
- 🐍 **Installation Issues**: Upgrade pip: `pip install --upgrade pip`.  
- 🗑️ **Corrupted Data**: Check `data_clean_report.csv`.  


## 🤝 Contributing
- Fork the repo and submit pull requests.  
- Report issues or suggest features via GitHub.  


## 🙏 Acknowledgments
- Dataset: Smoke Fire Detection YOLO Dataset (Kaggle)  
- Library: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- Inspired by various open-source object detection projects


## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Image Processing**
