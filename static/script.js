const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultImg = document.getElementById('result');

// Aktifkan webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

function captureAndSend() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image_data = canvas.toDataURL('image/jpeg');

  fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: image_data })
  })
  .then(res => res.json())
  .then(data => {
    if (data.image) {
      resultImg.src = data.image;
    }
  });
}

setInterval(captureAndSend, 1000); // kirim frame setiap 1 detik
