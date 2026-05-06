from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
from model import model_yoruglik, model_kontrast, model_neyron

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tahlil', methods=['POST'])
def tahlil():
    fayl = request.files['rasm']
    tech = request.form.get('tech', 'hammasi')
    
    rasm = Image.open(fayl).convert('L').resize((128, 128))
    arr = np.array(rasm)
    
    if tech == 'piksel':
        natija = [model_yoruglik(arr)]
    elif tech == 'gradient':
        natija = [model_kontrast(arr)]
    elif tech == 'neyron':
        natija = [model_neyron(arr)]
    else:
        natija = [model_yoruglik(arr), model_kontrast(arr), model_neyron(arr)]
    
    return jsonify(natija)

if __name__ == '__main__':
    app.run(debug=True)