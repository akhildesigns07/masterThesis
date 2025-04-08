const video = document.getElementById('video');
const caption = document.getElementById('caption');

// Request back camera
navigator.mediaDevices.getUserMedia({
  video: { facingMode: { ideal: "environment" } }
})
.then(stream => {
  video.srcObject = stream;
  captureAndSendFrame();
})
.catch(err => {
  console.error("Camera access error:", err);
  caption.textContent = "Could not access camera. Please allow camera permissions.";
});

function captureAndSendFrame() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = 320;
  canvas.height = 240;

  setInterval(() => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');

    fetch('/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
      caption.textContent = data.caption;
    })
    .catch(error => {
      console.error('Frame error:', error);
    });
  }, 2000);
}
