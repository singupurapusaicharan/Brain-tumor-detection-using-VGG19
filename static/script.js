const chooseFileBtn = document.querySelector('.choose-file-btn');
const fileInput = document.querySelector('input[type="file"]');
const outputImage = document.getElementById('output-image');

chooseFileBtn.addEventListener('click', () => {
  fileInput.click();
});

// Add event listeners for drag and drop functionality
const uploadArea = document.querySelector('.upload-area');

uploadArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadArea.classList.remove('drag-over');

  const file = event.dataTransfer.files[0];
  fileInput.files = event.dataTransfer.files;

  detectTumor(file);
});

// Function to detect tumors
function detectTumor(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
    const imageData = event.target.result;
    // Load TensorFlow.js model
    const model = tf.loadLayersModel('C:\\xampp\\htdocs\\register\\model\\vgg19_model.keras');
    model.predict(imageData).then(predictions => {
      // Display the output image
      // Update output image with the detected tumor
      const tumorPresent = predictions[0][0].toFixed(2);
      const predictedClass = predictions[0][1].toFixed(2);
      const tumorLocation = predictions[0][2].toFixed(2);

      // Send the prediction data to the Flask server
      fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/keras'
        },
        body: JSON.stringify({
          'tumor_present': tumorPresent,
          'predicted_class': predictedClass,
          'tumor_location': tumorLocation
        })
      })
      .then(response => response.keras())
      .then(data => {
        // Display the result
        outputImage.src = `data:image/jpeg;base64,${data['img_data']}`;
      });
    });
  };
  reader.readAsDataURL(file);
}