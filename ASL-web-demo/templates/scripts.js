document.getElementById('video-input').addEventListener('change', (event) => {
    const videoFile = event.target.files[0];

    if (videoFile) {
        const videoURL = URL.createObjectURL(videoFile);
        const originalVideoElement = document.getElementById('original-video');
        originalVideoElement.src = videoURL;

        originalVideoElement.addEventListener('canplay', () => {
            originalVideoElement.play();
        });
    }
});

document.getElementById('process-button').addEventListener('click', async () => {
    const videoFile = document.getElementById('video-input').files[0];
    const model = document.getElementById('model-select').value;

    if (!videoFile) {
        alert('Please upload a video file.');
        return;
    }

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('model', model);

    const response = await fetch('/process_video', { method: 'POST', body: formData });
    const result = await response.json();

    const processedVideoElement = document.getElementById('processed-video');
    processedVideoElement.src = result.processed_video_url;
    processedVideoElement.addEventListener('canplay', () => {
        processedVideoElement.play();
    });

    document.getElementById('prediction-result').textContent = result.prediction;
});

