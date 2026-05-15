# IH Cantabria – Coastal Flooding and Adaptation
 
A toolbox for coastal flooding assessment and climate adaptation analysis, developed by [IH Cantabria](https://github.com/IHCantabria).
 
---
 
## 📦 1. Installation
 
### 1.1 Clone the repository
 
Start by cloning the repository and navigating into the project folder:
 
```bash
git clone https://github.com/IHCantabria/IH-Cantabria-Coastal-Flooding-and-Adaptation.git
cd IH-Cantabria-Coastal-Flooding-and-Adaptation
```
 
---
 
### 1.2 Create an environment with Mamba/Anaconda
 
To run the toolbox you first need to install the required Python packages in an environment. It is highly recommended to use **Mamba**. Go to [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge) and download Miniforge for your operating system.
 
Once installed, open the **Miniforge Prompt** (on Mac/Linux, open a terminal), navigate to the project folder, and run the following commands:
 
```bash
mamba create -n coastal-flooding python=3.11 -y
mamba activate coastal-flooding
mamba install geopandas rasterio fiona contextily -y
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
 
> 💡 If you prefer Anaconda, replace `mamba` with `conda` in all commands.
 
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
jupyter lab
```
 
A browser window will open. Navigate to the repository folder and open the notebook. Run cells sequentially with **Shift + Enter**.
