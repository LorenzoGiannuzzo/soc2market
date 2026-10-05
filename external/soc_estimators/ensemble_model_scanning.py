from tensorflow import keras

# Carica il modello
model_path = r'C:\Users\lorenzo.giannuzzo\Documents\GitHub\Huawei_repo\SOC_prediction\nn_training\models\ensemble_dynamic_model.keras'

model = keras.models.load_model(model_path)

# Mostra architettura completa
print("=== MODEL SUMMARY ===")
model.summary()

print("\n=== LAYER DETAILS ===")
for layer in model.layers:
    try:
        config = layer.get_config()
        print(f"\n{layer.name} ({layer.__class__.__name__}):")
        if 'units' in config:
            print(f"  units: {config['units']}")
        if 'activation' in config:
            print(f"  activation: {config['activation']}")
        if 'rate' in config:  # Dropout
            print(f"  dropout rate: {config['rate']}")
    except:
        print(f"\n{layer.name}: (input layer)")