"""
================================================================================
SOC Prediction Framework for LFP Battery Cells
================================================================================

This work was originally developed in the context of the Huawei Tech Arena
Challenge (Nuremberg, 2024) for State of Charge (SOC) prediction of LFP
(Lithium Iron Phosphate) battery cells, where it was awarded First Prize.

The framework was subsequently extended and refined by researchers at the
Energy Center Lab, Department of Energy (DENERG), Politecnico di Torino.
The codebase, particularly the AI/Machine Learning model implementations,
was primarily developed by Lorenzo Giannuzzo.

Authors:
    Lorenzo Giannuzzo        - lorenzo.giannuzzo@polito.it
    Elena Rozzi               - elena.rozzi@polito.it
    Marcel Stolte             - marcel.stolte@polito.it
    Daniele Salvatore Schiera - daniele.schiera@polito.it
    Francesco D. Minuto       - francesco.minuto@polito.it
    Andrea Lanzini            - andrea.lanzini@polito.it

Supervision:
    Mansour Alramlawi         - Huawei Technologies
    (supervision of the related research paper and selected code modules)

Affiliation:
    Politecnico di Torino
    Department of Energy (DENERG) - Energy Center Lab
    Corso Duca degli Abruzzi 24, 10129 Torino, Italy

================================================================================
"""

# test_dense_layer.py - Esegui questo prima del retrain
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import numpy as np

# Build model with Dense(1) instead of Lambda
pred_input = layers.Input(shape=(6,), name='predictions')
context_input = layers.Input(shape=(9,), name='context')

x = layers.Dense(32, activation='relu')(context_input)
weights = layers.Dense(6, activation='softmax', name='weights')(x)
weighted = layers.Multiply()([pred_input, weights])

# THE FIX: Dense instead of Lambda
output = layers.Dense(1, use_bias=False, kernel_initializer='ones',
                      trainable=False, name='sum_output')(weighted)

model = Model(inputs=[pred_input, context_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Test save
model.save('test_model.keras')
print("✓ Saved")

# Test load (THIS IS WHERE IT FAILED BEFORE)
loaded = keras.models.load_model('test_model.keras')
print("✓ Loaded")

# Test predict
pred = loaded.predict([np.random.rand(1,6), np.random.rand(1,9)], verbose=0)
print(f"✓ Prediction: {pred[0,0]:.4f}")

print("\n✅ TEST PASSED - Il retrain funzionerà!")

# Cleanup
import os
os.remove('test_model.keras')