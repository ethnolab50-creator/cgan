/**
 * CGAN Web Application - Frontend JavaScript
 * Handles user interactions and API communication
 */

class CGANApp {
    constructor() {
        this.apiBase = window.location.origin;
        this.isTraining = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStatus();
    }

    bindEvents() {
        // Training controls
        document.getElementById('train-btn').addEventListener('click', () => this.trainModel());
        document.getElementById('load-btn').addEventListener('click', () => this.loadModel());

        // Generation controls
        document.getElementById('generate-btn').addEventListener('click', () => this.generateDigit());
        document.getElementById('generate-all-btn').addEventListener('click', () => this.generateAllDigits());

        // Enter key support
        document.getElementById('epochs').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.trainModel();
        });
    }

    async updateStatus() {
        try {
            const response = await fetch(`${this.apiBase}/api/status`);
            const data = await response.json();

            document.getElementById('status-text').textContent =
                data.trained ? 'Trained' : 'Not trained';
            document.getElementById('device-text').textContent = data.device || 'Unknown';

            // Update button states
            this.updateButtonStates(data.trained);

        } catch (error) {
            console.error('Error updating status:', error);
        }
    }

    updateButtonStates(isTrained) {
        const generateBtn = document.getElementById('generate-btn');
        const generateAllBtn = document.getElementById('generate-all-btn');

        generateBtn.disabled = !isTrained;
        generateAllBtn.disabled = !isTrained;

        if (isTrained) {
            generateBtn.classList.remove('disabled');
            generateAllBtn.classList.remove('disabled');
        } else {
            generateBtn.classList.add('disabled');
            generateAllBtn.classList.add('disabled');
        }
    }

    async trainModel() {
        if (this.isTraining) return;

        const epochs = parseInt(document.getElementById('epochs').value);
        if (epochs < 1 || epochs > 50) {
            this.showAlert('Please enter epochs between 1 and 50');
            return;
        }

        this.isTraining = true;
        this.showTrainingModal();

        try {
            const response = await fetch(`${this.apiBase}/api/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ epochs: epochs })
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.showAlert(`Training completed successfully! Final losses - D: ${result.final_losses.d_loss.toFixed(4)}, G: ${result.final_losses.g_loss.toFixed(4)}`, 'success');
                this.updateStatus();
            } else {
                this.showAlert(`Training failed: ${result.message}`, 'error');
            }

        } catch (error) {
            console.error('Training error:', error);
            this.showAlert('Training failed due to network error', 'error');
        } finally {
            this.isTraining = false;
            this.hideTrainingModal();
        }
    }

    async loadModel() {
        this.showLoadingModal('Loading model...');

        try {
            const response = await fetch(`${this.apiBase}/api/load_model`, {
                method: 'POST'
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.showAlert('Model loaded successfully!', 'success');
                this.updateStatus();
            } else {
                this.showAlert(`Failed to load model: ${result.message}`, 'error');
            }

        } catch (error) {
            console.error('Load error:', error);
            this.showAlert('Failed to load model', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async generateDigit() {
        const digit = parseInt(document.getElementById('digit-select').value);
        const numSamples = parseInt(document.getElementById('num-samples').value);

        this.showLoadingModal('Generating digits...');

        try {
            const response = await fetch(`${this.apiBase}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    digit: digit,
                    num_samples: numSamples
                })
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.displayGeneratedImage(result.image, `Digit ${digit}`);
            } else {
                this.showAlert(`Generation failed: ${result.message}`, 'error');
            }

        } catch (error) {
            console.error('Generation error:', error);
            this.showAlert('Generation failed due to network error', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    async generateAllDigits() {
        this.showLoadingModal('Generating all digits...');

        try {
            const response = await fetch(`${this.apiBase}/api/generate_all`);
            const result = await response.json();

            if (result.status === 'success') {
                this.displayAllGeneratedImages(result.digits);
            } else {
                this.showAlert(`Generation failed: ${result.message}`, 'error');
            }

        } catch (error) {
            console.error('Generation error:', error);
            this.showAlert('Generation failed due to network error', 'error');
        } finally {
            this.hideLoadingModal();
        }
    }

    displayGeneratedImage(imageData, title) {
        const container = document.getElementById('results-container');

        // Clear previous results
        container.innerHTML = '';

        // Create image element
        const imageItem = document.createElement('div');
        imageItem.className = 'image-item';

        const img = document.createElement('img');
        img.src = imageData;
        img.alt = title;

        const titleElement = document.createElement('h3');
        titleElement.textContent = title;

        imageItem.appendChild(img);
        imageItem.appendChild(titleElement);
        container.appendChild(imageItem);
    }

    displayAllGeneratedImages(digitsData) {
        const container = document.getElementById('results-container');

        // Clear previous results
        container.innerHTML = '';

        // Create grid container
        const grid = document.createElement('div');
        grid.className = 'generated-images';

        // Add each digit
        for (let digit = 0; digit <= 9; digit++) {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';

            const img = document.createElement('img');
            img.src = digitsData[digit.toString()];
            img.alt = `Digit ${digit}`;

            const titleElement = document.createElement('h3');
            titleElement.textContent = `Digit ${digit}`;

            imageItem.appendChild(img);
            imageItem.appendChild(titleElement);
            grid.appendChild(imageItem);
        }

        container.appendChild(grid);
    }

    showTrainingModal() {
        const modal = document.getElementById('training-modal');
        const progressFill = document.getElementById('progress-fill');
        const trainingInfo = document.getElementById('training-info');

        modal.style.display = 'block';
        progressFill.style.width = '0%';
        trainingInfo.textContent = 'Initializing training...';

        // Simulate progress (in real implementation, this would come from server)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 100) progress = 100;
            progressFill.style.width = `${progress}%`;
            trainingInfo.textContent = `Training... ${Math.round(progress)}% complete`;

            if (progress >= 100) {
                clearInterval(progressInterval);
            }
        }, 500);
    }

    hideTrainingModal() {
        document.getElementById('training-modal').style.display = 'none';
    }

    showLoadingModal(message) {
        const modal = document.getElementById('loading-modal');
        document.getElementById('loading-text').textContent = message;
        modal.style.display = 'block';
    }

    hideLoadingModal() {
        document.getElementById('loading-modal').style.display = 'none';
    }

    showAlert(message, type = 'info') {
        // Simple alert - in production, you might want a more sophisticated notification system
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;
        alertDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            z-index: 1001;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;

        // Set background color based on type
        const colors = {
            'success': '#4CAF50',
            'error': '#f44336',
            'warning': '#ff9800',
            'info': '#2196F3'
        };
        alertDiv.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(alertDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cganApp = new CGANApp();
});