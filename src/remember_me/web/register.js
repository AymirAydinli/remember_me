const maxImageSize = 10 * 1024 * 1024;

const form = document.querySelector("#registration-form");
const imageInput = document.querySelector("#image");
const previewImage = document.querySelector("#preview-image");
const previewPlaceholder = document.querySelector("#preview-placeholder");
const submitButton = document.querySelector("#register-button");
const formMessage = document.querySelector("#form-message");

let previewUrl = null;

function showMessage(message, state = "") {
  formMessage.textContent = message;
  formMessage.className = `form-message ${state}`.trim();
}

function clearPreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewImage.hidden = true;
  previewImage.removeAttribute("src");
  previewPlaceholder.hidden = false;
}

imageInput.addEventListener("change", () => {
  clearPreview();
  const image = imageInput.files?.[0];

  if (!image) {
    return;
  }

  if (image.size > maxImageSize) {
    imageInput.value = "";
    showMessage("The image must be smaller than 10 MB.", "error");
    return;
  }

  if (!image.type.startsWith("image/")) {
    imageInput.value = "";
    showMessage("Choose a valid image file.", "error");
    return;
  }

  previewUrl = URL.createObjectURL(image);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  showMessage("Ready to register this person.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const data = new FormData(form);
  const image = imageInput.files?.[0];

  if (!image) {
    showMessage("Choose a face photo.", "error");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Processing...";
  showMessage("Detecting the face and creating an embedding...");

  try {
    const response = await fetch("/people/register", {
      method: "POST",
      body: data,
    });
    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(body.detail || "Registration failed.");
    }

    showMessage(
      `${body.name} was registered as ${body.relationship}.`,
      "success"
    );
    form.reset();
    clearPreview();
  } catch (error) {
    showMessage(error.message || "Registration failed.", "error");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Register person";
  }
});

window.addEventListener("beforeunload", clearPreview);
