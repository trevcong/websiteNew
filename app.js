document.addEventListener('DOMContentLoaded', () => {
    const authService = new AuthService();
    
    // Get the current page name
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    // Handle auth state changes globally
    firebase.auth().onAuthStateChanged(user => {
        if (user) {
            // User is logged in
            updateUIForLoggedInUser(user);
        } else {
            // User is logged out
            updateUIForLoggedOutUser();
        }
    });

    // Function to update UI for logged-in user
    function updateUIForLoggedInUser(user) {
        // Update navigation if it exists
        const loginBtn = document.querySelector('.login-btn');
        if (loginBtn) {
            loginBtn.textContent = 'ML Demo';
            loginBtn.href = 'ml-demo.html';
        }

        // Handle page-specific logic
        switch (currentPage) {
            case 'login.html':
                // Redirect to ML demo if already logged in
                window.location.href = 'ml-demo.html';
                break;

            case 'ml-demo.html':
                // Update ML demo page UI
                document.getElementById('userEmail').textContent = user.email;
                setupMLDemoPage();
                break;
        }
    }

    // Function to update UI for logged-out user
    function updateUIForLoggedOutUser() {
        // Update navigation if it exists
        const loginBtn = document.querySelector('.login-btn');
        if (loginBtn) {
            loginBtn.textContent = 'ML Demo Login';
            loginBtn.href = 'login.html';
        }

        // Handle page-specific logic
        if (currentPage === 'ml-demo.html') {
            // Redirect to login if trying to access ML demo while logged out
            window.location.href = 'login.html';
        }
    }

    // Set up event listeners based on current page
    switch (currentPage) {
        case 'login.html':
            setupLoginPage();
            break;
        case 'ml-demo.html':
            setupMLDemoPage();
            break;
        case 'index.html':
            setupHomePage();
            break;
    }

    // Setup functions for each page
    function setupLoginPage() {
        const loginForm = document.getElementById('loginForm');
        const googleLoginBtn = document.getElementById('googleLogin');
        const showSignupLink = document.getElementById('showSignup');

        if (loginForm) {
            loginForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const email = document.getElementById('loginEmail').value;
                const password = document.getElementById('loginPassword').value;
                await authService.signIn(email, password);
            });
        }

        if (googleLoginBtn) {
            googleLoginBtn.addEventListener('click', () => {
                authService.signInWithGoogle();
            });
        }

        if (showSignupLink) {
            showSignupLink.addEventListener('click', (e) => {
                e.preventDefault();
                // Handle signup request - you can add your own logic here
                alert('Please contact the administrator for demo access.');
            });
        }
    }

    function setupMLDemoPage() {
        const logoutBtn = document.getElementById('logoutBtn');
        const AUTHORIZED_EMAILS = [
            'your-email@example.com',
            // Add more authorized emails
        ];

        // Check authorization
        authService.checkPageAuthorization(AUTHORIZED_EMAILS).then(isAuthorized => {
            if (!isAuthorized) {
                window.location.href = 'unauthorized.html';
                return;
            }
        });

        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                authService.signOut().then(() => {
                    window.location.href = 'index.html';
                });
            });
        }

        // Add your ML demo specific initialization here
        // For example:
        initializeMLDemo();
    }

    function setupHomePage() {
        // Add any home page specific initialization here
        // For example: animations, portfolio filters, etc.
    }

    // Your ML demo initialization function
    // website/app.js
function initializeMLDemo() {
    // DOM Elements
    const elements = {
        video: document.getElementById('videoElement'),
        verifyVideo: document.getElementById('verifyVideoElement'),
        fileInput: document.getElementById('fileInput'),
        verifyFileInput: document.getElementById('verifyFileInput'),
        imagePreview: document.getElementById('imagePreview'),
        verifyImagePreview: document.getElementById('verifyImagePreview'),
        uploadButton: document.getElementById('uploadButton'),
        verifyUploadButton: document.getElementById('verifyUploadButton'),
        dropZone: document.getElementById('dropZone'),
        trainButton: document.getElementById('trainButton'),
        verifyButton: document.getElementById('verifyButton'),
        switchCamera: document.getElementById('switchCamera'),
        verifySwitchCamera: document.getElementById('verifySwitchCamera'),
        captureButton: document.getElementById('captureButton'),
        loadingDiv: document.getElementById('loading'),
        verifyLoadingDiv: document.getElementById('verifyLoading'),
        resultContainer: document.getElementById('resultContainer'),
        tabButtons: document.querySelectorAll('.tab-button'),
        tabContents: document.querySelectorAll('.tab-content'),
        trainingGallery: document.getElementById('trainingGallery')
    };

    let currentStream = null;
    let verifyStream = null;
    let facingMode = 'user';
    let selectedFiles = new Set();

    // Initialize cameras
    initializeCamera(elements.video, { current: currentStream });
    initializeCamera(elements.verifyVideo, { current: verifyStream });

    // Tab Switching
    function switchTab(tabId) {
        elements.tabButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        elements.tabContents.forEach(content => {
            content.classList.toggle('active', content.id === `${tabId}Tab`);
        });

        if (tabId === 'gallery') {
            loadTrainingGallery();
        }
    }

    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            switchTab(tabId);
        });
    });

    // File Handling Functions
    function handleFileSelection(e) {
        const files = Array.from(e.target.files);
        handleFiles(files);
    }

    function handleFiles(files) {
        files.forEach(file => {
            if (file.type.startsWith('image/')) {
                selectedFiles.add(file);
                previewImage(file, elements.imagePreview);
            }
        });
        updateTrainButtonState();
    }

    function previewImage(file, previewElement) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewElement.src = e.target.result;
            previewElement.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Camera Functions
    async function initializeCamera(videoElement, streamSetter) {
        try {
            if (streamSetter.current) {
                streamSetter.current.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    facingMode: facingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = stream;
            streamSetter.current = stream;
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Error accessing camera. Please make sure you have granted camera permissions.');
        }
    }

    function captureFrame(videoElement) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext('2d').drawImage(videoElement, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.8);
    }

    // API Interaction Functions
    async function trainFace(imageData) {
        const response = await fetch('http://localhost:5000/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                userId: firebase.auth().currentUser.uid,
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error('Training failed');
        }

        return response.json();
    }

    async function verifyFace(imageData) {
        const response = await fetch('http://localhost:5000/api/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                userId: firebase.auth().currentUser.uid,
                image: imageData
            })
        });

        if (!response.ok) {
            throw new Error('Verification failed');
        }

        return response.json();
    }

    // Gallery Functions
    async function loadTrainingGallery() {
        elements.trainingGallery.innerHTML = '';
        const userId = firebase.auth().currentUser.uid;
        
        try {
            const response = await fetch(`http://localhost:5000/api/trained-images/${userId}`);
            const images = await response.json();
            
            images.forEach(image => {
                const imageElement = createGalleryImage(image);
                elements.trainingGallery.appendChild(imageElement);
            });
        } catch (error) {
            console.error('Error loading gallery:', error);
        }
    }

    function createGalleryImage(imageData) {
        const container = document.createElement('div');
        container.className = 'trained-image';
        
        const img = document.createElement('img');
        img.src = imageData.url;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => removeTrainedImage(imageData.id);
        
        container.appendChild(img);
        container.appendChild(removeBtn);
        return container;
    }

    async function removeTrainedImage(imageId) {
        try {
            await fetch(`http://localhost:5000/api/trained-images/${imageId}`, {
                method: 'DELETE',
            });
            loadTrainingGallery();
        } catch (error) {
            console.error('Error removing image:', error);
        }
    }

    // Event Listeners
    elements.fileInput.addEventListener('change', handleFileSelection);
    elements.verifyFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            previewImage(file, elements.verifyImagePreview);
        }
    });

    elements.uploadButton.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.verifyUploadButton.addEventListener('click', () => {
        elements.verifyFileInput.click();
    });

    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('dragover');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('dragover');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.type.startsWith('image/')
        );
        handleFiles(files);
    });

    elements.switchCamera.addEventListener('click', () => {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        initializeCamera(elements.video, { current: currentStream });
    });

    elements.verifySwitchCamera.addEventListener('click', () => {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        initializeCamera(elements.verifyVideo, { current: verifyStream });
    });

    elements.captureButton.addEventListener('click', () => {
        const imageData = captureFrame(elements.video);
        const blob = dataURItoBlob(imageData);
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        selectedFiles.add(file);
        elements.imagePreview.src = imageData;
        elements.imagePreview.style.display = 'block';
        updateTrainButtonState();
    });

    elements.trainButton.addEventListener('click', async () => {
        elements.loadingDiv.style.display = 'flex';
        elements.trainButton.disabled = true;

        try {
            for (const file of selectedFiles) {
                const imageData = await fileToDataUrl(file);
                await trainFace(imageData);
            }

            elements.resultContainer.className = 'result-container match';
            elements.resultContainer.textContent = 'Training completed successfully!';
            selectedFiles.clear();
            elements.imagePreview.style.display = 'none';
            updateTrainButtonState();
            loadTrainingGallery();
        } catch (error) {
            elements.resultContainer.className = 'result-container no-match';
            elements.resultContainer.textContent = `Training failed: ${error.message}`;
        } finally {
            elements.loadingDiv.style.display = 'none';
            elements.trainButton.disabled = false;
        }
    });

    elements.verifyButton.addEventListener('click', async () => {
        elements.verifyLoadingDiv.style.display = 'flex';
        elements.verifyButton.disabled = true;

        try {
            const imageData = elements.verifyImagePreview.src || captureFrame(elements.verifyVideo);
            const result = await verifyFace(imageData);

            elements.resultContainer.className = `result-container ${result.match ? 'match' : 'no-match'}`;
            elements.resultContainer.textContent = result.match
                ? `Match found! Confidence: ${(result.confidence * 100).toFixed(1)}%`
                : 'No match found';
        } catch (error) {
            elements.resultContainer.className = 'result-container no-match';
            elements.resultContainer.textContent = `Verification failed: ${error.message}`;
        } finally {
            elements.verifyLoadingDiv.style.display = 'none';
            elements.verifyButton.disabled = false;
        }
    });

    // Utility Functions
    function updateTrainButtonState() {
        elements.trainButton.disabled = selectedFiles.size === 0;
    }

    function dataURItoBlob(dataURI) {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }

        return new Blob([ab], { type: mimeString });
    }

    function fileToDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsDataURL(file);
        });
    }
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const imagePreview = document.getElementById('imagePreview');

    // Make the entire drop zone clickable
    dropZone.addEventListener('click', (e) => {
        // Prevent click from triggering twice when clicking the button
        if (e.target !== uploadButton) {
            fileInput.click();
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            handleFiles(files);
        }
    });

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.type.startsWith('image/')
        );
        
        if (files.length > 0) {
            handleFiles(files);
        }
    });

    // Handle the files
    function handleFiles(files) {
        // Clear previous preview
        imagePreview.style.display = 'none';
        
        // Store the files
        selectedFiles = new Set(files);
        
        // Show preview of the first file
        if (files.length > 0) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(files[0]);
        }
        
        // Update UI to show number of selected files
        const fileCount = files.length;
        const fileText = fileCount === 1 ? '1 file selected' : `${fileCount} files selected`;
        uploadButton.textContent = fileText;
        
        // Enable/disable train button
        updateTrainButtonState();
    }

    // Update button state
    function updateTrainButtonState() {
        const trainButton = document.getElementById('trainButton');
        trainButton.disabled = selectedFiles.size === 0;
    }

    // Prevent default button click behavior
    uploadButton.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent event from bubbling to dropZone
        fileInput.click();
    });

    // Preview multiple images
    function previewImages(files) {
        const previewContainer = document.querySelector('.preview-container');
        previewContainer.innerHTML = ''; // Clear previous previews

        files.forEach(file => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('preview-image');
                previewContainer.appendChild(img);
            };
            reader.readAsDataURL(file);
        });
    }

    // Add this CSS for multiple image previews
    const style = document.createElement('style');
    style.textContent = `
        .preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
        .preview-image {
            max-width: 150px;
            max-height: 150px;
            border-radius: 4px;
            object-fit: cover;
        }
    `;
    document.head.appendChild(style);

    // Initialize
    updateTrainButtonState();
}
    // Error handling
    window.onerror = function(msg, url, lineNo, columnNo, error) {
        console.error('Error: ', msg, '\nURL: ', url, '\nLine: ', lineNo, '\nColumn: ', columnNo, '\nError object: ', error);
        return false;
    };


});