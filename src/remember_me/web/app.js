const recognitionInterval = 2500;
const unknownConfirmationFrames = 2;

const video = document.querySelector("#camera");
const cameraFrame = document.querySelector("#camera-frame");
const canvas = document.querySelector("#capture-canvas");
const faceBox = document.querySelector("#face-box");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const details = document.querySelector("#person-details");
const statusText = document.querySelector("#recognition-status");
const personName = document.querySelector("#person-name");
const personRelationship = document.querySelector(
  "#person-relationship"
);

let cameraStream = null;
let recognitionTimer = null;
let requestController = null;
let lastFace = null;
let consecutiveUnknownResults = 0;

function updateDetails(state, status, name, relationship = "") {
  details.className = state;
  statusText.textContent = status;
  personName.textContent = name;
  personRelationship.textContent = relationship;
}

function hideFaceBox() {
  lastFace = null;
  faceBox.hidden = true;
  faceBox.className = "";
}

function positionFaceBox(face, state) {
  if (
    !face ||
    !video.videoWidth ||
    !video.videoHeight
  ) {
    hideFaceBox();
    return;
  }

  lastFace = face;

  const frameWidth = cameraFrame.clientWidth;
  const frameHeight = cameraFrame.clientHeight;

  const scale = Math.min(
    frameWidth / video.videoWidth,
    frameHeight / video.videoHeight
  );

  const displayedWidth = video.videoWidth * scale;
  const displayedHeight = video.videoHeight * scale;
  const offsetX = (frameWidth - displayedWidth) / 2;
  const offsetY = (frameHeight - displayedHeight) / 2;

  faceBox.style.left = `${offsetX + face.x * scale}px`;
  faceBox.style.top = `${offsetY + face.y * scale}px`;
  faceBox.style.width = `${face.width * scale}px`;
  faceBox.style.height = `${face.height * scale}px`;

  faceBox.className = state;
  faceBox.hidden = false;
}

function scheduleRecognition(delay = recognitionInterval) {
  clearTimeout(recognitionTimer);

  if (cameraStream) {
    recognitionTimer = setTimeout(recognizeFrame, delay);
  }
}

function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) {
    return Promise.resolve(null);
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.85);
  });
}

async function recognizeFrame() {
  if (!cameraStream) {
    return;
  }

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
      consecutiveUnknownResults = 0;
      hideFaceBox();
      updateDetails(
        "",
        "Looking",
        body.detail || "No face detected"
      );
      return;
    }

    if (!response.ok) {
      throw new Error(body.detail || "Recognition failed");
    }

    if (body.recognized) {
      consecutiveUnknownResults = 0;
      positionFaceBox(body.face, "recognized");
      updateDetails(
        "recognized",
        "Recognized",
        body.name,
        body.relationship
      );
    } else {
      consecutiveUnknownResults += 1;

      if (consecutiveUnknownResults < unknownConfirmationFrames) {
        positionFaceBox(body.face, "scanning");
        updateDetails(
          "scanning",
          "Scanning",
          "Checking this face..."
        );
      } else {
        positionFaceBox(body.face, "unknown");
        updateDetails(
          "unknown",
          "Unknown",
          "Unknown person"
        );
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      consecutiveUnknownResults = 0;
      hideFaceBox();
      updateDetails(
        "error",
        "Error",
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
  consecutiveUnknownResults = 0;

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

    cameraFrame.classList.add("active");
    stopButton.disabled = false;

    updateDetails("", "Looking", "No person detected");
    scheduleRecognition(0);
  } catch {
    cameraStream = null;
    startButton.disabled = false;

    updateDetails(
      "error",
      "Camera unavailable",
      "Allow camera access and try again"
    );
  }
}

function stopCamera() {
  clearTimeout(recognitionTimer);
  requestController?.abort();
  consecutiveUnknownResults = 0;

  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;

  cameraFrame.classList.remove("active");
  startButton.disabled = false;
  stopButton.disabled = true;

  hideFaceBox();
  updateDetails("", "Camera is off", "No person detected");
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

window.addEventListener("resize", () => {
  if (lastFace && !faceBox.hidden) {
    const state = faceBox.className;
    positionFaceBox(lastFace, state);
  }
});

window.addEventListener("beforeunload", stopCamera);
