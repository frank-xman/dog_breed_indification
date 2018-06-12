# dog_breed_indification
---
the statement of this dog_breeds:  
  1.this is a baidu data, not kaggle, but the model can both use;  
  2.100 breeds of dog;  
  3.use the single_model can reach the acc=0.77, and the mix model can be better  
'''
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(1024,activation='relu')(x)
predictions = Dense(100,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
'''  
use the modol of inception(googlenet) single model to train the data.
