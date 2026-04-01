import urllib.request

if not os.path.exists('model.pkl'):
    print("Downloading model...")
    urllib.request.urlretrieve(
        'https://huggingface.co/rithiksai/fraud-detection-model/resolve/main/model.pkl',
        'model.pkl'
    )
    print("Model downloaded! ✅")

if not os.path.exists('scaler.pkl'):
    print("Downloading scaler...")
    urllib.request.urlretrieve(
        'https://drive.google.com/uc?id=1rbeqZJhRlZNPGWt-urQ0kYqrIRLX9HyD&export=download',
        'scaler.pkl'
    )
    print("Scaler downloaded! ✅")