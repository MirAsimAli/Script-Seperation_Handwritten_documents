# ✍️ Indic Script Separation Models (Handwritten Documents)

🚀 A machine learning project for **word-level script separation** in Indic **handwritten documents**, developed as part of an OCR pipeline at **IIIT Hyderabad**.  
This repo demonstrates the **testing workflow** for handwritten text recognition with strong focus on clarity, reproducibility, and visualization.  

---

## 📖 Overview
- ✅ Developed **13 bilingual** and **12 multilingual** ML models.  
- ✅ Based on **ResNet18 architectures** for handwritten data.  
- ✅ Full ML pipeline: *data curation → preprocessing → training → validation → testing → deployment*.  
- ✅ Integrated into the **IIIT Hyderabad iLock site** for real-world OCR usage.  
- ✅ Only the **test script** + **README** are shared here (training scripts & weights may not be shareable).  

> ⚠️ *Note*: Internal datasets and training weights are not included. Repo is meant to showcase the **testing setup**.

---

## 🛠️ Tech Stack  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet18-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/CSV-000000?style=for-the-badge&logo=csv&logoColor=white"/>
  <img src="https://img.shields.io/badge/API%20Integration-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white"/>
</p>

---

## 📂 Repository Structure
```bash
Indic-Script-Separation-Handwritten/
│── test_handwritten.py       # 🔬 Test script for evaluating handwritten models
│── README.md                 # 📘 Project documentation
│── results/                  # 📊 Confusion matrix & loss curves (auto-generated)
```

---

## 🚀 How to Run  
```bash
# 1️⃣ Clone the repository  
git clone https://github.com/YourUsername/Indic-Script-Separation-Handwritten.git  
cd Indic-Script-Separation-Handwritten  

# 2️⃣ Install dependencies  
pip install -r requirements.txt  

# 3️⃣ Run the test script  
python test_handwritten.py  
```

---

## 📊 Visualizations  

### 🔹 Confusion Matrix  
<p align="center">
  <img src="results/confusion_matrix_example.png" alt="Confusion Matrix" width="500"/>
</p>

### 🔹 Loss Curves  
<p align="center">
  <img src="results/loss_curves_example.png" alt="Loss Curves" width="500"/>
</p>

---

## 👨‍💻 Author  
**Asim Ali**  
Machine Learning Engineer (Intern) at IIIT Hyderabad  
