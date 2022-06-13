# Dermanalyze-ML

[Go to Parent Project](https://github.com/tenpoless/dermanalyze)

## Data 
Our model is built to predict 8 skin cancer and non skin cancer labels. Here are the labels we use:
* `akiec`: actinic keratoses and intraepithelial carcinoma / bowen's disease
* `bcc`: basal cell carcinoma
* `bkl`: benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)
* `df`: dermatofibroma
* `mel`: melanoma
* `nv`: melanocytic nevi
* `vasc`: vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)
* `unk`: normal skin (include mole) and other skin diseases (acne, milia)

## Data Extraction
We have been collecting and retrieving disparate types of data from a variety of sources.
Then, we created the directories structure. In these folders we stored train and validation images that will later be fed to the Keras generators.

## Data Exploration
Through data exploration, we view characteristics of the dataset. Next, we plot images that have the same lesion_id to see that the duplicate lesions have the same image. We also view data distribution in each label and the result is the number of images in each label differs greatly, this indicates that the data is not balanced, which is imbalanced data.

## Data Preprocessing
Before the model is trained or used to predict, the image to be used will be pre-processed using `shades of gray color constancy`. Shades of gray color constancy are useful for constanting the image, especially on images of reddish skin and images taken using a dermascope. This is very helpful in the training process because the data sets we use come from different sources, where the way their photos are taken will also vary. So by using shades of gray color constancy, the model will focus on training/predicting the lesion, regardless of illuminations and skin condition.

## Modeling
We used the `EfficientNet B6` model and did `fine tuning` on it. The EfficientNet B6 model was chosen because it has better performance than the other models we tested using the tuner. Fine tuning is done by freezing 2/3 of the model and training the remaining 1/3 of it.

## Evaluation
We used `balanced accuracy` as the metric and `categorical cross-entropy` as the loss function, where the loss function used during training is weighted categorical cross-entropy. Weight is calculated using N/n formula, where n is the number of data in a class and N is the total number of data. We choose them as evaluators because we have imbalanced data, where label nv has more than five thousand data, while other classes only have hundreds of data. After 20 epochs, our model reaches 0.4904 balanced_accuracy, 23.8055 weighted loss, 0.4906 val_balanced_accuracy, and 3.8012 val_loss.

Please take a note, we only train the model with the label unk (unknown) without validating it. This is done to prevent overfit, where we don't know exactly what images will come out as unk (unknown).

## Deployment
We chose to deploy the model using a server, but to reduce the large file size, we optimized it by converting the model to tflite. We don't quentize it to prevent long inference process. We managed to reduce the model size to 4x smaller with default optimization, without reducing the performance of the model.

## Model Reproducibility
1. Download and unzip [dataset](https://drive.google.com/drive/folders/1esYey7I0c82KqloE1NhkTW5SPGgbnL5t?usp=sharing)
2. Upgrade pip
    ```
    python -m pip install ––upgrade pip
    ```
3. Install requirements
    ```
    pip install -r requirements.txt
    ```
4. Run `Dermanalyze.ipynb` on import, defining path, data augmentation and its bottom blocks
5. Make sure dataset is in the same directory as the Dermanalyze.ipynb file
6. You can do inference by using this [test set](https://drive.google.com/drive/folders/1IwmbGWkDLde5KQHbPQse2yc8ytbXNhAZ?usp=sharing)

## References
### Dataset
* [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* [ISIC Archive 2018](https://challenge.isic-archive.com/data/#2018)
* [Skin Lesions Dermatoscopic Images](https://www.kaggle.com/datasets/jnegrini/skin-lesions-act-keratosis-and-melanoma)
* [DermNet NZ](https://dermnetnz.org/image-library)
* [Normal Skin 1](https://ijdvl.com/the-utility-of-dermoscopy-in-the-diagnosis-of-evolving-lesions-of-vitiligo/)
* [Normal Skin 2](https://www.researchgate.net/publication/263710653_In-vivo_imaging_of_psoriatic_lesions_with_polarization_multispectral_dermoscopy_and_multiphoton_microscopy)
* [Normal Skin 3](https://www.researchgate.net/publication/220451508_Systematic_design_of_a_cross-polarized_dermoscope_for_visual_inspection_and_digital_imaging)
* [Normal Skin 4](https://www.researchgate.net/publication/270658281_Detecting_melanoma_in_dermoscopy_images_using_scale_adaptive_local_binary_patterns)
* [Normal Skin 5](https://www.ijdpdd.com/article.asp?issn=2349-6029;year=2017;volume=4;issue=2;spage=27;epage=30;aulast=Nirmal)
* [Normal Skin 6](https://www.semanticscholar.org/paper/Dermoscopy-in-near-full-facial-transplantation.-Kami%C5%84ska-Winciorek-Giebel/54f1a4de702261cafaf5fbdf129f1d2326650b92#paper-header)
* [Normal Skin 7](http://www.odermatol.com/issue-in-html/2018-2-34-nevus/)
### Journal
* [C. Barata, M. E. Celebi, and J. S. Marques, “Improving dermoscopy image classification using color constancy,” IEEE journal of biomedical and health informatics, vol. 19, no. 3, pp. 1146–1152, 2015.](https://faculty.uca.edu/ecelebi/documents/JBHI_2015.pdf)
