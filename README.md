# OrderFusion
Orderbook Feature Learning and Asymmetric Generalization in Intraday Electricity Markets

🦊 Summary page: https://runyao-yu.github.io/AsymGen/

🌋 Paper link: https://arxiv.org/pdf/placeholder

![Description of Image](Figure/static/images/Phases.PNG)


---


## 🚀 Quick Start

We open-source all code for preprocessing, modeling, and analysis.  
The project directory is structured as follows:

    OrderFusion/
    ├── Data/
        |- Country (e.g. Germany)
            |- Intraday Continuous
                |- Orders
                    |- Year (e.g. 2023)
                        |- Month (e.g. 01)
                        |- Month (e.g. 02)
                        |- Month (e.g. 03)
                        ...
                    ...
    ├── Figure/
    ├── Result/
    ├── Orderbook_Preprocessing.py
    ├── Orderbook_Preprocessing.ipynb
    ├── Asymmetric_Generalization.py
    ├── Asymmetric_Generalization.ipynb
    ├── README.md

The file `README.md` specifies the required package versions.

### ✅ Step 1: Prepare the Folder Structure
Place the purchased orderbook data into `Data` folder. Purchase source: https://webshop.eex-group.com/epex-spot-public-market-data (Several data types are available. For example, the “Continuous
Anonymous Orders History” for Germany costs 325 EUR/month.)

### ✅ Step 2: Feature and Label Extraction

Run `Orderbook_Preprocessing.ipynb` to extract input features and output labels

The script `Orderbook_Preprocessing.py` contains all necessary functions and classes.

### ✅ Step 3: Replication of Asymmetric Generalization Phenomenon

Run `Asymmetric_Generalization.ipynb` to conduct various transfer learning experiments

The script `Asymmetric_Generalization.py` contains all necessary functions and classes.

### ✅ Other Information

Inside `Result` folder: 

- the `agg_....csv` reveals the detailed aggregated feature importance per category. 

- the `top_features_....csv` reveals the ranking of features through feature selection.

---


## 📦 Environment & Dependencies

This project has been tested with the following environment:

- **Python 3.9.20**
- `numpy==1.25.2`
- `pandas==2.1.4`
- `scikit-learn==1.5.1`
- `tensorflow==2.16.2`
- `protobuf>=3.19.0`
- `h5py>=3.1.0`
- `joblib`
- `setuptools`
- `tqdm`
- `natsort`

Use the following comment to pip install:

```bash
pip install numpy==1.25.2 pandas==2.1.4 scikit-learn==1.5.1 scipy==1.13.1 tensorflow==2.16.2 protobuf>=3.19.0 h5py>=3.1.0 joblib setuptools tqdm natsort

