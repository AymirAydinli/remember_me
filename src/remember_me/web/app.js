const recognitionInterval = 2500;
const unknownConfirmationFrames = 2;
const maxRecordingDuration = 2 * 60 * 1000;

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
const conversationControls = document.querySelector(
  "#conversation-controls"
);
const startRecordingButton = document.querySelector("#start-recording");
const stopRecordingButton = document.querySelector("#stop-recording");
const recordingStatus = document.querySelector("#recording-status");

let cameraStream = null;
let recognitionTimer = null;
let requestController = null;
let lastFace = null;
let consecutiveUnknownResults = 0;
let currentPersonId = null;
let recordingPersonId = null;
let audioStream = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimeout = null;
let conversationBusy = false;

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

  if (cameraStream && !conversationBusy) {
    recognitionTimer = setTimeout(recognizeFrame, delay);
  }
}

function showConversationControls(personId) {
  currentPersonId = personId;
  conversationControls.hidden = false;

  if (!conversationBusy) {
    startRecordingButton.disabled = false;
  }
}

function clearConversationControls() {
  if (conversationBusy) {
    return;
  }

  currentPersonId = null;
  conversationControls.hidden = true;
  startRecordingButton.disabled = false;
  stopRecordingButton.disabled = true;
  recordingStatus.textContent = "";
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
      clearConversationControls();
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
      showConversationControls(body.person_id);
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
        clearConversationControls();
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
      clearConversationControls();
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

  if (mediaRecorder?.state === "recording") {
    stopRecording();
  }

  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.srcObject = null;

  cameraFrame.classList.remove("active");
  startButton.disabled = false;
  stopButton.disabled = true;

  hideFaceBox();
  clearConversationControls();
  updateDetails("", "Camera is off", "No person detected");
}

function preferredAudioType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function releaseAudioStream() {
  audioStream?.getTracks().forEach((track) => track.stop());
  audioStream = null;
}

async function uploadRecording(audioBlob, contentType, personId) {
  const formData = new FormData();
  const extension = contentType.startsWith("audio/mp4") ? "mp4" : "webm";
  formData.append("audio", audioBlob, `conversation.${extension}`);
  recordingStatus.textContent = "Transcribing and summarizing...";

  try {
    const response = await fetch(`/api/people/${personId}/conversations`, {
      method: "POST",
      body: formData,
    });
    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(body.detail || "Conversation processing failed");
    }

    recordingStatus.textContent = "Conversation summary saved.";
  } catch (error) {
    recordingStatus.textContent = error.message || "Conversation could not be saved.";
  } finally {
    conversationBusy = false;
    recordingPersonId = null;
    stopRecordingButton.disabled = true;

    if (cameraStream) {
      startRecordingButton.disabled = currentPersonId === null;
      scheduleRecognition(0);
    } else {
      clearConversationControls();
    }
  }
}

async function finishRecording() {
  clearTimeout(recordingTimeout);
  recordingTimeout = null;
  releaseAudioStream();

  const contentType = mediaRecorder?.mimeType || audioChunks[0]?.type || "audio/webm";
  const audioBlob = new Blob(audioChunks, { type: contentType });
  const personId = recordingPersonId;
  mediaRecorder = null;
  audioChunks = [];

  if (!audioBlob.size || personId === null) {
    conversationBusy = false;
    recordingPersonId = null;
    recordingStatus.textContent = "No audio was recorded.";
    startRecordingButton.disabled = currentPersonId === null;
    stopRecordingButton.disabled = true;
    scheduleRecognition(0);
    return;
  }

  await uploadRecording(audioBlob, contentType, personId);
}

async function startRecording() {
  if (currentPersonId === null || conversationBusy) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    recordingStatus.textContent = "Audio recording is not supported in this browser.";
    return;
  }

  conversationBusy = true;
  recordingPersonId = currentPersonId;
  clearTimeout(recognitionTimer);
  requestController?.abort();
  startRecordingButton.disabled = true;

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = preferredAudioType();
    mediaRecorder = mimeType
      ? new MediaRecorder(audioStream, { mimeType })
      : new MediaRecorder(audioStream);
    audioChunks = [];

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });
    mediaRecorder.addEventListener("stop", finishRecording, { once: true });
    mediaRecorder.start();

    stopRecordingButton.disabled = false;
    recordingStatus.textContent = "Recording conversation...";
    recordingTimeout = setTimeout(stopRecording, maxRecordingDuration);
  } catch {
    releaseAudioStream();
    conversationBusy = false;
    recordingPersonId = null;
    startRecordingButton.disabled = false;
    stopRecordingButton.disabled = true;
    recordingStatus.textContent = "Microphone permission is required.";
    scheduleRecognition(0);
  }
}

function stopRecording() {
  clearTimeout(recordingTimeout);
  recordingTimeout = null;
  stopRecordingButton.disabled = true;

  if (mediaRecorder?.state === "recording") {
    recordingStatus.textContent = "Preparing recording...";
    mediaRecorder.stop();
  }
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
startRecordingButton.addEventListener("click", startRecording);
stopRecordingButton.addEventListener("click", stopRecording);

window.addEventListener("resize", () => {
  if (lastFace && !faceBox.hidden) {
    const state = faceBox.className;
    positionFaceBox(lastFace, state);
  }
});

window.addEventListener("beforeunload", stopCamera);
