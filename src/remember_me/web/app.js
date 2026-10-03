const recognitionInterval = 2500;

const video = document.querySelector("#camera");
const cameraFrame = document.querySelector(".camera-frame");
const canvas = document.querySelector("#capture-canvas");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const result = document.querySelector("#recognition-result");
const statusLabel = document.querySelector("#status-label");
const personName = document.querySelector("#person-name");
const relationship = document.querySelector("#person-relationship");

let cameraStream = null;
let recognitionTimer = null;
let requestController = null;

function updateResult(state, status, name, relation = "") {
  result.dataset.state = state;
  statusLabel.textContent = status;
  personName.textContent = name;
  relationship.textContent = relation;
}

function scheduleRecognition(delay = recognitionInterval) {
  clearTimeout(recognitionTimer);

  if (cameraStream) {
    recognitionTimer = setTimeout(recognizeCurrentFrame, delay);
  }
}

function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) {
    return null;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.85);
  });
}

async function recognizeCurrentFrame() {
  if (!cameraStream) {
    return;
  }

  updateResult(
    "searching",
    "Looking",
    "Checking for a familiar person..."
  );

  try {
    const image = await captureFrame();

    if (!image || !cameraStream) {
      return;
    }

    const formData = new FormData();
    formData.append("image", image, "webcam-frame.jpg");

    requestController = new AbortController();

    const response = await fetch("/api/recognize", {
      method: "POST",
      body: formData,
      signal: requestController.signal,
    });

    const body = await response.json();

    if (response.status === 422) {
      updateResult(
        "searching",
        "Looking",
        body.detail || "Position one face in the camera"
      );
      return;
    }

    if (!response.ok) {
      throw new Error(body.detail || "Recognition failed");
    }

    if (body.recognized) {
      updateResult(
        "recognized",
        "Familiar person",
        body.name,
        body.relationship
      );
    } else {
      updateResult(
        "unknown",
        "Unknown person",
        "I do not recognize this person"
      );
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      updateResult(
        "error",
        "Camera assistance unavailable",
        error.message
      );
    }
  } finally {
    requestController = null;
    scheduleRecognition();
  }
}

async function startCamera() {
  startButton.disabled = true;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = cameraStream;
    await video.play();

    cameraFrame.classList.add("is-active");
    stopButton.disabled = false;

    updateResult(
      "searching",
      "Camera active",
      "Looking for a familiar person..."
    );

    scheduleRecognition(0);
  } catch (error) {
    cameraStream = null;
    startButton.disabled = false;

    updateResult(
      "error",
      "Camera unavailable",
      "Allow camera access and try again"
    );
  }
}

function stopCamera() {
  clearTimeout(recognitionTimer);
  requestController?.abort();

  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;

  cameraFrame.classList.remove("is-active");
  startButton.disabled = false;
  stopButton.disabled = true;

  updateResult(
    "idle",
    "Camera is off",
    "No recognition in progress"
  );
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("beforeunload", stopCamera);