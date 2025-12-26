# City-Transformers

City-Transformers is a framework for estimating **building construction age** from **street-view imagery (SVI)** by classifying building facades into four construction-epoch bins:

* **Class 1:** pre-1940
* **Class 2:** 1941–1970
* **Class 3:** 1971–1990
* **Class 4:** post-1990

The repository implements two deep learning approaches:

* **CSWin-Transformer** – a hierarchical vision transformer with cross-shaped window attention
* **SimCLR (ResNet-50 backbone)** – a contrastive learning framework fine-tuned for classification

Pretrained weights for both models are included.

The repository also contains a **Google Street View (GSV) scraping notebook** used to build:

* an **out-of-distribution (OOD)** test set, and
* a **target set** for properties missing imagery in the city’s database.

Human evaluation data can be made available **upon reasonable request**.

---

## 📂 Data

### **Training / Validation Data**

Training data comes from the Worcester city open-data housing facade dataset, hosted on Hugging Face:

👉 **[https://huggingface.co/datasets/murai-lab/WorcesterMA_Housing_Facades](https://huggingface.co/datasets/murai-lab/WorcesterMA_Housing_Facades)**

This dataset includes:

* 22,949 labeled facade images
* Labels mapped to the 4 construction-era classes above

### **Out-of-Distribution (OOD) Test Data**

OOD data must be collected **via Google Street View**, using:

```
worcester-web-scraper/gsv_scraper.ipynb
```

You will need:

* a Google Street View API key
* to comply with Google's Terms of Service

This notebook also builds the **target set** for properties with no images in the city dataset.

### **Pretrained Weights**

Pretrained weights for CSWin and SimCLR models are available in:

```
models/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/murai-lab/City-Transformers.git
cd City-Transformers
```

Ensure you have:

* Python 3.8+
* PyTorch (with CUDA if running on GPU)

---

## 🚀 Training

### ** Prepare the Data**

Download and extract the Hugging Face dataset.
Training scripts expect the following structure:

```
abap/
    train/
    val/
    metadata.csv
```

Set the MVIT_PATH environment variable to your path:

```
export MVIT_PATH=/path/to/data/abap/ 
```

## 🌐 Scraping Google Street View (GSV)

The notebook:

```
worcester-web-scraper/gsv_scraper.ipynb
```

allows you to:

* Provide a list of property IDs or addresses
* Fetch corresponding GSV images
* Save them locally for evaluation or target-set creation

⚠️ **Important:**
You must use your own GSV API key and adhere to Google Street View **Terms of Service**.

The training scripts will automatically replace `abap` by `gsv` in the MVIT_PATH variable. Hence, the GSV Dataset folder (`gsv`) must be located on the same directory as the City Housing Dataset (`abap`) folder.

---

## 📑 Citation

If you use this repository, please cite:

### **Dataset**

```
@misc{murai2025worcester_dataset,
  author       = {Shannon Song and Yiqing Zhang and Fabricio Murai and Nan Ma},
  title        = {WorcesterMA_Housing_Facades Dataset},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/murai-lab/WorcesterMA_Housing_Facades}}
}
```

---

## 📜 License

This project is released under the **MIT License**.
See the `LICENSE` file for details.
