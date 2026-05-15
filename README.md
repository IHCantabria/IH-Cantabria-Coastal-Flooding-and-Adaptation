 # IH Cantabria – Coastal Flooding and Adaptation
 
A toolbox for coastal flooding assessment and climate adaptation analysis, developed by [IH Cantabria](https://github.com/IHCantabria).
 
---
 
## 📦 1. Installation
 
### 1.1 Install Miniforge
 
To manage the Python environment, it is highly recommended to use **Mamba**. Go to [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge) and download Miniforge for your operating system. Follow the installer instructions.
 
> 💡 If you prefer Anaconda, you can use it instead and replace `mamba` with `conda` in all commands below.
 
---
 
### 1.2 Clone the repository
 
Open the **Miniforge Prompt** (on Mac/Linux, open a terminal). Navigate to the folder where you want to download the project and clone the repository:
 
```bash
cd path/to/your/folder
git clone https://github.com/IHCantabria/IH-Cantabria-Coastal-Flooding-and-Adaptation.git
cd IH-Cantabria-Coastal-Flooding-and-Adaptation
```
 
---
 
### 1.3 Create the environment and install dependencies
 
From inside the project folder, run the following commands to create the environment and install all required packages:
 
```bash
mamba create -n coastal-flooding python=3.11 -y
mamba activate coastal-flooding
mamba install geopandas rasterio fiona contextily jupyterlab -y
pip install -r requirements.txt
```
 
All required packages are now installed in a self-contained environment called `coastal-flooding`. Always activate it before working:
 
```bash
mamba activate coastal-flooding
```
 
Your terminal prompt should now start with `(coastal-flooding)`.
 
> ⚠️ **If errors are raised during installation**, clean up and retry:
> ```bash
> mamba clean --all
> mamba update conda
> ```
 
---
 
## 🚀 2. Usage
 
The main entry point is the Jupyter Notebook:
 
📓 **`IH Cantabria - Coastal Flooding and Adaptation.ipynb`**
 
To launch it, open the Miniforge Prompt, activate the environment, and start JupyterLab:
 
```bash
mamba activate coastal-flooding
jupyter lab
```
 
A browser window will open. Navigate to the repository folder and open the notebook. Run cells sequentially with **Shift + Enter**.
```
 
All required packages are now installed in a self-contained environment called `coastal-flooding`. Always activate it before working:
 
```bash
mamba activate coastal-flooding
```
 
Your terminal prompt should now start with `(coastal-flooding)`.
 
> ⚠️ **If errors are raised during installation**, clean up and retry:
> ```bash
> mamba clean --all
> mamba update conda
> ```
 
---
 
## 🚀 2. Usage
 
The main entry point is the Jupyter Notebook:
 
📓 **`IH Cantabria - Coastal Flooding and Adaptation.ipynb`**
 
To launch it, open the Miniforge Prompt, activate the environment, and start JupyterLab:
 
```bash
mamba activate coastal-flooding
jupyter lab
```
 
A browser window will open. Navigate to the repository folder and open the notebook. Run cells sequentially with **Shift + Enter**.
 
A browser window will open. Navigate to the repository folder and open the notebook. Run cells sequentially with **Shift + Enter**.
