model = create_model()

model.fit(text_ds, verbose=2, epochs=25, callbacks=[text_gen_callback])
