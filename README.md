> # dog_breed_indification
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
								target_size=(299,299),  
								batch_size=48)

###### this is a problem i met: if i transform this data into the xxx.npy this will be too big:   
   
 ### this 3 pics is use the ImageDataGenerator to generator to train the model

 ![the_pic_1](https://github.com/frank-xman/dog_breed_indification/blob/master/data/2_0_2665.jpg)
 ![the_pic_2](https://github.com/frank-xman/dog_breed_indification/blob/master/data/2_0_614.jpg)
 ![the_pic_3](https://github.com/frank-xman/dog_breed_indification/blob/master/data/2_0_8728.jpg)

this pic of data is show in the data filefold, (not all, just some of them, if someone interest in it can go to the baiduyunpan:)
  
  
	def lr_decay(epoch):# this code is main to adjust the learning_rate with the epochs change
	    lrs = [0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000001, 0.0000001]
	    return lrs[epoch]
	    
##### inception-v3-single_model finally achieve the acc=0.77   
if use mix model the resnet and the xception, the train time will decrease and the acc will increase;
this part i m ding now, maybe some time will commit the code and the result.
##### ps: i really need a marchine can cal the big data.................
the code of train_data is show at the [train_data.py](https://github.com/frank-xman/dog_breed_indification/blob/master/train_data.py)

and this is the train pic:  
![train](https://github.com/frank-xman/dog_breed_indification/blob/master/training.jpg)
  
### res=
![res](https://github.com/frank-xman/dog_breed_indification/blob/master/res_acc.jpg)


	    



	
