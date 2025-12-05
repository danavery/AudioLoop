// AudioLoop Web Labeling Interface
// Simple vanilla JavaScript - no frameworks needed

class AudioLabelingInterface {
    constructor() {
        this.currentIndex = 0;
        this.totalCandidates = 0;
        this.isLoaded = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.setupKeyboardShortcuts();
    }
    
    initializeElements() {
        // Get references to DOM elements
        this.elements = {
            // Forms and inputs
            loadForm: document.getElementById('loadForm'),
            candidatesFile: document.getElementById('candidatesFile'),
            dataset: document.getElementById('dataset'),
            audioDir: document.getElementById('audioDir'),
            
            // Main sections
            loadingSection: document.getElementById('loadingSection'),
            mainContent: document.getElementById('mainContent'),
            
            // Audio player
            audioPlayer: document.getElementById('audioPlayer'),
            playBtn: document.getElementById('playBtn'),
            stopBtn: document.getElementById('stopBtn'),
            
            // Labeling controls
            positiveBtn: document.getElementById('positiveBtn'),
            negativeBtn: document.getElementById('negativeBtn'),
            
            // Navigation
            prevBtn: document.getElementById('prevBtn'),
            nextBtn: document.getElementById('nextBtn'),
            nextUnlabeledBtn: document.getElementById('nextUnlabeledBtn'),
            jumpInput: document.getElementById('jumpInput'),
            jumpBtn: document.getElementById('jumpBtn'),
            
            // Info display
            candidateInfo: document.getElementById('candidateInfo'),
            currentLabel: document.getElementById('currentLabel'),
            modelPrediction: document.getElementById('modelPrediction'),
            progressBar: document.getElementById('progressBar'),
            progressDetails: document.getElementById('progressDetails'),
            progressText: document.getElementById('progressText'),
            
            // Actions
            saveBtn: document.getElementById('saveBtn')
        };
    }
    
    attachEventListeners() {
        // Form submission
        this.elements.loadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.loadCandidates();
        });
        
        // Audio controls
        this.elements.playBtn.addEventListener('click', () => this.playAudio());
        this.elements.stopBtn.addEventListener('click', () => this.stopAudio());
        
        // Labeling buttons
        this.elements.positiveBtn.addEventListener('click', () => this.labelCurrent('1'));
        this.elements.negativeBtn.addEventListener('click', () => this.labelCurrent('0'));
        
        // Navigation
        this.elements.prevBtn.addEventListener('click', () => this.navigateTo(this.currentIndex - 1));
        this.elements.nextBtn.addEventListener('click', () => this.navigateTo(this.currentIndex + 1));
        this.elements.nextUnlabeledBtn.addEventListener('click', () => this.jumpToNextUnlabeled());
        this.elements.jumpBtn.addEventListener('click', () => this.jumpToSample());
        
        // Save
        this.elements.saveBtn.addEventListener('click', () => this.saveLabels());
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (!this.isLoaded) return;
            
            // Ignore keystrokes when typing in input fields
            if (e.target.tagName === 'INPUT') return;
            
            switch(e.key) {
                case '1':
                case 'y':
                case 'Y':
                    e.preventDefault();
                    this.labelCurrent('1');
                    break;
                case '0':
                case 'n':
                case 'N':
                    e.preventDefault();
                    this.labelCurrent('0');
                    break;
                case 'p':
                case 'P':
                case ' ':
                    e.preventDefault();
                    this.playAudio();
                    break;
                case 'x':
                case 'X':
                    e.preventDefault();
                    this.stopAudio();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigateTo(this.currentIndex - 1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigateTo(this.currentIndex + 1);
                    break;
                case 'u':
                case 'U':
                    e.preventDefault();
                    this.jumpToNextUnlabeled();
                    break;
                case 's':
                case 'S':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.saveLabels();
                    }
                    break;
            }
        });
    }
    
    async loadCandidates() {
        const candidatesFile = this.elements.candidatesFile.value.trim();
        const dataset = this.elements.dataset.value;
        const audioDir = this.elements.audioDir.value.trim() || null;
        
        if (!candidatesFile) {
            this.showError('Please provide a candidates CSV file path');
            return;
        }
        
        try {
            const response = await fetch('/api/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    candidates_csv: candidatesFile,
                    dataset: dataset,
                    audio_dir: audioDir
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showError(result.error);
                return;
            }
            
            // Success - switch to main interface
            this.totalCandidates = result.total_candidates;
            this.isLoaded = true;

            this.elements.loadingSection.style.display = 'none';
            this.elements.mainContent.style.display = 'flex';
            
            // Enable controls
            this.enableControls();
            
            // Load first candidate
            this.currentIndex = 0;
            await this.loadCurrentCandidate();
            
            this.showSuccess(`Loaded ${result.total_candidates} candidates (${result.labeled_count} already labeled)`);
            
        } catch (error) {
            this.showError('Failed to load candidates: ' + error.message);
        }
    }
    
    async loadCurrentCandidate() {
        try {
            const response = await fetch(`/api/candidate/${this.currentIndex}`);
            const result = await response.json();
            
            if (result.error) {
                this.showError(result.error);
                return;
            }
            
            // Update UI with candidate info
            this.updateCandidateDisplay(result);
            this.updateProgress(result);
            
            // Update audio source
            this.elements.audioPlayer.src = `/api/audio/${this.currentIndex}`;
            
            // Auto-play audio
            this.playAudio();
            
        } catch (error) {
            this.showError('Failed to load candidate: ' + error.message);
        }
    }
    
    updateCandidateDisplay(result) {
        const candidate = result.candidate;
        
        // Update candidate info
        this.elements.candidateInfo.innerHTML = `
            <p><strong>Sample:</strong> ${result.index + 1} of ${result.total}</p>
            <p><strong>Filename:</strong> ${candidate.filename}</p>
            <p><strong>Prediction:</strong> ${candidate.prediction}</p>
            <p><strong>Confidence:</strong> ${candidate.confidence}</p>
            <p><strong>Original Class:</strong> ${candidate.original_class}</p>
        `;
        
        // Update current label display
        if (candidate.current_label) {
            const labelText = candidate.current_label === '1' ? 'Positive (1)' : 'Negative (0)';
            const labelClass = candidate.current_label === '1' ? 'bg-success' : 'bg-danger';
            this.elements.currentLabel.textContent = labelText;
            this.elements.currentLabel.className = `badge fs-6 ${labelClass}`;
        } else {
            this.elements.currentLabel.textContent = 'Not yet labeled';
            this.elements.currentLabel.className = 'badge bg-secondary fs-6';
        }
        
        // Update model prediction
        this.elements.modelPrediction.textContent = candidate.prediction;
        
        // Update jump input max
        this.elements.jumpInput.max = result.total;
        this.elements.jumpInput.value = result.index + 1;
    }
    
    updateProgress(result) {
        const progressPercent = (result.labeled_count / result.total) * 100;
        this.elements.progressBar.style.width = `${progressPercent}%`;
        this.elements.progressBar.textContent = `${progressPercent.toFixed(1)}%`;
        this.elements.progressDetails.textContent = `${result.labeled_count} / ${result.total} labeled`;
        this.elements.progressText.textContent = `Sample ${result.index + 1} of ${result.total}`;
        
        // Update navigation button states
        this.elements.prevBtn.disabled = this.currentIndex === 0;
        this.elements.nextBtn.disabled = this.currentIndex === result.total - 1;
    }
    
    async labelCurrent(label) {
        try {
            const response = await fetch('/api/label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    index: this.currentIndex,
                    label: label
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showError(result.error);
                return;
            }
            
            // Success - update display and auto-advance
            const labelText = label === '1' ? 'Positive' : 'Negative';
            this.showSuccess(`Labeled as ${labelText}`);
            
            // Auto-advance to next sample
            if (this.currentIndex < this.totalCandidates - 1) {
                await this.navigateTo(this.currentIndex + 1);
            } else {
                // Refresh current to show updated label
                await this.loadCurrentCandidate();
            }
            
        } catch (error) {
            this.showError('Failed to save label: ' + error.message);
        }
    }
    
    async navigateTo(index) {
        if (index < 0 || index >= this.totalCandidates) {
            return;
        }
        
        this.currentIndex = index;
        await this.loadCurrentCandidate();
    }
    
    async jumpToNextUnlabeled() {
        try {
            const response = await fetch(`/api/next_unlabeled/${this.currentIndex + 1}`);
            const result = await response.json();
            
            if (result.next_index !== null) {
                await this.navigateTo(result.next_index);
            } else {
                this.showInfo('All samples have been labeled!');
            }
            
        } catch (error) {
            this.showError('Failed to find next unlabeled: ' + error.message);
        }
    }
    
    async jumpToSample() {
        const sampleNum = parseInt(this.elements.jumpInput.value);
        if (sampleNum >= 1 && sampleNum <= this.totalCandidates) {
            await this.navigateTo(sampleNum - 1);
        }
    }
    
    playAudio() {
        this.elements.audioPlayer.currentTime = 0;
        this.elements.audioPlayer.play().catch(e => {
            console.warn('Audio play failed:', e);
        });
    }
    
    stopAudio() {
        this.elements.audioPlayer.pause();
        this.elements.audioPlayer.currentTime = 0;
    }
    
    async saveLabels() {
        try {
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showError(result.error);
                return;
            }
            
            this.showSuccess('Labels saved successfully!');
            
        } catch (error) {
            this.showError('Failed to save labels: ' + error.message);
        }
    }
    
    enableControls() {
        const controls = [
            'positiveBtn', 'negativeBtn', 'prevBtn', 'nextBtn', 
            'nextUnlabeledBtn', 'jumpBtn', 'saveBtn'
        ];
        
        controls.forEach(id => {
            if (this.elements[id]) {
                this.elements[id].disabled = false;
            }
        });
    }
    
    showSuccess(message) {
        this.showAlert(message, 'success');
    }
    
    showError(message) {
        this.showAlert(message, 'danger');
    }
    
    showInfo(message) {
        this.showAlert(message, 'info');
    }
    
    showAlert(message, type) {
        // Simple toast-like notification
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 3000);
    }
}

// Initialize the interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AudioLabelingInterface();
});