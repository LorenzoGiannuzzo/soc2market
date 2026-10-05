from tensorflow import keras

model = keras.models.load_model('nn_training/models/ensemble_selector_model.keras')

print("=== INPUTS ===")
for i, inp in enumerate(model.inputs):
    print(f"  Input {i}: {inp.name} - shape {inp.shape}")

print("\n=== LAYERS ===")
for layer in model.layers:
    try:
        print(f"  {layer.name}: {layer.output.shape}")
    except:
        print(f"  {layer.name}: (input layer)")

print("\n=== MODEL SUMMARY ===")
model.summary()