"""
CGAN Web Application Backend
============================

Flask server for CGAN (Conditional GAN) demonstration with MNIST digits.
Provides REST API for training CGAN and generating conditional digit samples.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import json
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for frontend requests

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = None
discriminator = None
train_loader = None
is_trained = False
training_history = []

# CGAN Model Architecture for MNIST
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        img_flat = self.model(gen_input)
        return img_flat.view(img_flat.size(0), 1, 28, 28)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_embedding = self.label_emb(labels)
        disc_input = torch.cat([img_flat, label_embedding], dim=1)
        return self.model(disc_input)

# Helper Functions
def load_data(batch_size=64):
    """Load MNIST dataset"""
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        return train_loader
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def initialize_models(latent_dim=100, num_classes=10):
    """Initialize generator and discriminator models"""
    generator = ConditionalGenerator(latent_dim, num_classes).to(device)
    discriminator = ConditionalDiscriminator(num_classes).to(device)
    return generator, discriminator

def train_cgan_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion):
    """Train CGAN for one epoch"""
    generator.train()
    discriminator.train()

    total_d_loss = 0
    total_g_loss = 0
    num_batches = 0

    for real_images, labels in train_loader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_d.zero_grad()

        # Real images
        real_output = discriminator(real_images, labels)
        d_loss_real = criterion(real_output, real_labels)

        # Fake images
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = generator(noise, labels)
        fake_output = discriminator(fake_images.detach(), labels)
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        fake_output = discriminator(fake_images, labels)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        num_batches += 1

    return total_d_loss / num_batches, total_g_loss / num_batches

def generate_digit_images(generator, digit, num_samples=5):
    """Generate images for a specific digit"""
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        fake_images = generator(noise, labels)

        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        fake_images = fake_images.cpu().numpy()

    return fake_images

def create_digit_grid(images, digit):
    """Create a grid image from generated samples"""
    num_samples = images.shape[0]

    # Calculate grid dimensions
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        ax.imshow(images[i, 0], cmap='gray')
        ax.axis('off')

    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        ax.axis('off')

    plt.suptitle(f'Generated Digit: {digit}', fontsize=16)
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)

    return buf

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return '', 204

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the CGAN model"""
    global generator, discriminator, is_trained, training_history

    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 5)

        # Initialize models and data
        train_loader = load_data()
        if train_loader is None:
            return jsonify({'error': 'Failed to load training data'}), 500

        generator, discriminator = initialize_models()

        # Optimizers and loss
        optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        training_history = []

        for epoch in range(epochs):
            d_loss, g_loss = train_cgan_epoch(generator, discriminator, train_loader,
                                             optimizer_g, optimizer_d, criterion)

            training_history.append({
                'epoch': epoch + 1,
                'd_loss': float(d_loss),
                'g_loss': float(g_loss)
            })

            print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        is_trained = True

        # Save models
        torch.save(generator.state_dict(), 'cgan_generator.pth')
        torch.save(discriminator.state_dict(), 'cgan_discriminator.pth')

        return jsonify({
            'status': 'success',
            'message': f'Training completed for {epochs} epochs',
            'epochs': epochs,
            'training_history': training_history,
            'final_losses': training_history[-1]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_samples():
    """Generate digit images for a specific number"""
    global generator, is_trained

    try:
        if not is_trained or generator is None:
            return jsonify({'status': 'error', 'message': 'Model not trained yet. Please train first.'}), 400

        data = request.get_json() or {}
        digit = data.get('digit', 0)
        num_samples = data.get('num_samples', 5)

        if not (0 <= digit <= 9):
            return jsonify({'status': 'error', 'message': 'Digit must be between 0 and 9'}), 400

        # Generate images
        images = generate_digit_images(generator, digit, num_samples)

        # Create grid image
        buf = create_digit_grid(images, digit)

        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'digit': digit,
            'num_samples': num_samples,
            'image': f'data:image/png;base64,{img_base64}'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate_all', methods=['GET'])
def generate_all_samples():
    """Generate samples for all digits (0-9)"""
    global generator, is_trained

    try:
        if not is_trained or generator is None:
            return jsonify({'status': 'error', 'message': 'Model not trained yet. Please train first.'}), 400

        results = {}

        for digit in range(10):
            images = generate_digit_images(generator, digit, 3)  # 3 samples each
            buf = create_digit_grid(images, digit)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            results[str(digit)] = f'data:image/png;base64,{img_base64}'

        return jsonify({
            'status': 'success',
            'message': 'Generated samples for all digits',
            'digits': results
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get training and model status"""
    global is_trained, training_history

    return jsonify({
        'trained': is_trained,
        'device': str(device),
        'training_history': training_history,
        'model_files': {
            'generator': os.path.exists('cgan_generator.pth'),
            'discriminator': os.path.exists('cgan_discriminator.pth')
        }
    })

@app.route('/api/load_model', methods=['POST'])
def load_saved_model():
    """Load saved model weights"""
    global generator, discriminator, is_trained

    try:
        if not (os.path.exists('cgan_generator.pth') and os.path.exists('cgan_discriminator.pth')):
            return jsonify({'status': 'error', 'message': 'Model files not found'}), 404

        generator, discriminator = initialize_models()

        generator.load_state_dict(torch.load('cgan_generator.pth', map_location=device))
        discriminator.load_state_dict(torch.load('cgan_discriminator.pth', map_location=device))

        is_trained = True

        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available digit classes (0-9)"""
    classes = []
    for i in range(10):
        classes.append({
            'id': i,
            'name': f'Digit {i}',
            'description': f'Generate handwritten digit {i}'
        })
    return jsonify({'classes': classes})

if __name__ == '__main__':
    print("Starting CGAN Web Application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)