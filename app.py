from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms as transforms
from mynet import predict_image, preprocess_image
from user_manager import register_user, validate_user, class_dict, class_descriptions

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


def get_model(path="./models/best.pth", num_of_category=10):
    net = resnet50()
    net.fc = nn.Linear(net.fc.in_features, num_of_category)
    net.load_state_dict(torch.load(path))
    return net


net = get_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        register_user(username, password)
        flash('User registered successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if validate_user(username, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')


@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    class_id, confidence = predict_image(file_path, net)
    class_id += 1
    print(f"id: {class_id}")
    class_name = class_dict[str(class_id)]
    description = class_descriptions[str(class_id)]
    return render_template('result.html', filename=filename, class_name=class_name, confidence=confidence,
                           description=description)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
