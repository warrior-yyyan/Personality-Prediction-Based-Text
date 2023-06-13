This project is mainly used for text-based personality prediction tasks, mainly based on the pre-training model + classifier model, and currently provides data processing support for two datasets, Essays and Kaggle. Of course, you can add your own dataset, and the specific processing logic can refer to the code.

If you want to use this project to predict personality, you should follow the steps below:
1. View `extract_features.py` and set the address of the local pre-trained model. This file is mainly used to extract features.
2. If you added your own dataset, please edit your own **Dataset** in `datasets.py`.
3. `classify.py` is the specific training process, please check the relevant parameters and set the required file path before use.


Our project is based on `Python 3.8` and `Pytorch 1.6`. Of course, since we are using the basic library, the version change will not affect it.
