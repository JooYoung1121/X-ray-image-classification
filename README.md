# Recognition of the presence of a lung lesion in an X-ray image using deep learning

# 1. requirement

tensorflow-gpu, numpy, pandas, opencv, keras, scikit-learning, etc


# 2. data_download


kaggle chest x_ray data download -> (1. kaggle api 2. original download)



# 3. data_preprocess 


112000 data set -> 8:1:1 slicing image -> tensor


# 4. training 


python final_train.py  -> VGG19 + STN + CNN

python std_train.py -> Vanillia CNN 

# 5. test

python test.py

