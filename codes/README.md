### Transformers
##### ESM transformer
To use the transformers, in the exact form of their embedding, the data_provider_esm.py file is prepared for each of the two types of dataset.
If you want to use this, follow these steps:
1) install esm with this command !pip install git+https://github.com/facebookresearch/esm.git
2) In the train.py and test.py files, change from dataprovider to from data_provider_esm_iedb or data_provider_esm_anthem.
3) In the model.py file, class Model, comment self.encoder_hla_a2 = CNN_HLA_Encoder(23) and uncomment elf.encoder_hla_a2 = CNN_HLA_Encoder_esm(1280, config.batch_size)
##### ESMFold transformer (Structure Prediction)
To use the transformers, in the exact form of their embedding, the data_provider_esmfold.py file is prepared for each of the two types of dataset.
If you want to use this, follow these steps:
1) In the train.py and test.py files, change from dataprovider to from data_provider_esmfold_iedb or data_provider_esmfold_anthem.
2) In the model.py file, class Model, comment self.encoder_hla_a2 = CNN_HLA_Encoder(23) and uncomment self.encoder_hla_a2 = CNN_HLA_Encoder_esmfold(366)
### Hyperparameter tuning
In this work we use Ray Tuning for tune network hyperparameter. For tuning at first !pip install ray and then run !python 'run_tuning.py' Anthem_test/config.json.
