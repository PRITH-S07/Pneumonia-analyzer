from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint('Pneumonia_classification.h5',
                             monitor='val_loss', 
                             mode='min',
                             save_best_only=True,
                             verbose=1)
callbacks = [checkpoint]
epochs=10
history = model.fit(x=train_generator,epochs = epochs,validation_data = validation_generator,callbacks=callbacks)
