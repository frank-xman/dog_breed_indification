# dog_breed_indification
---
the statement of this dog_breeds:  
  1.this is a baidu data, not kaggle, but the model can both use;  
  2.100 breeds of dog;  
  3.use the single_model can reach the acc=0.77, and the mix model can be better  
    
	base_model = InceptionV3(weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x) 
	x = Dense(1024,activation='relu')(x)
	predictions = Dense(100,activation='softmax')(Dropout(0.5)(x))
	model = Model(inputs=base_model.input,outputs=predictions)
  
use the modol of inception(googlenet) single model to train the data.  
and cause the data is relative small, so use the argument to increase the data: 

		train_datagen=ImageDataGenerator(    
			rescale=1./255,
			shear_range=0.2,
			width_shift_range=0.3,
			height_shift_range=0.3,
			rotation_range=30,
			zoom_range=0.3,
			horizontal_flip=True,
			vertical_flip=True)
		train_generator = train_datagen.flow_from_directory(directory='train1/',
										  target_size=(299,299),  # inception V3 image_size, the resnet is (224,224)
										  batch_size=48)

##### this is a problem i met: if i transform this data into the xxx.npy this will be too big:  


	
