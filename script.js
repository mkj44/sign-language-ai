let video = document.getElementById("webcam");
let overlay = document.getElementById("overlay");
let ctx = overlay.getContext("2d");

let startBtn = document.getElementById("startBtn");
let stopBtn = document.getElementById("stopBtn");
let loadModelBtn = document.getElementById("loadModelBtn");
let speakToggle = document.getElementById("speakToggle");
let cameraStatusDiv = document.getElementById("cameraStatus"); // New element

let labelDiv = document.getElementById("label");
let confidenceBar = document.getElementById("confidenceBar");
let confidenceText = document.getElementById("confidenceText");
let spokenTranscription = document.getElementById("spokenTranscription"); // New element

let textToSpeakArea = document.getElementById("textToSpeak"); // New element
let speakTextBtn = document.getElementById("speakTextBtn"); // New element

let model = null;
let labels = [];
let running = false;
let rafId = null;
let lastDisplayedLabel = "";

// --- Core Functionality ---

// Start webcam with status update
async function startCamera() {
  cameraStatusDiv.textContent = "Camera: Starting...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    cameraStatusDiv.textContent = "Camera: Active ✅";
  } catch (e) {
    alert("Could not start camera. Please allow camera permission.");
    console.error(e);
    cameraStatusDiv.textContent = "Camera: Failed ❌";
  }
}

// Load model and labels
async function loadModel() {
  loadModelBtn.disabled = true;
  loadModelBtn.textContent = "Loading...";
  try {
    // Note: Use the public path or relative path to your model files
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded!");
    const metadata = await fetch("model/metadata.json").then(res => res.json());
    // Fallback labels for safety
    labels = metadata.labels || ["hello", "yes", "no", "help"];
    alert("Model loaded successfully!");
    loadModelBtn.textContent = "Model Loaded";
    loadModelBtn.disabled = true; // Keep disabled once loaded
    startBtn.disabled = false; // Enable start button after model is ready
  } catch (e) {
    console.error("Error loading model:", e);
    alert("Failed to load model. Check console for details.");
    loadModelBtn.textContent = "Load Model (Failed)";
    loadModelBtn.disabled = false;
  }
}

// Speak and transcribe the output
function speakAndTranscribe(text) {
  // Update transcription for deaf users
  spokenTranscription.textContent = text;
  
  if (!speakToggle.checked) return;
  if (!("speechSynthesis" in window)) return;
  
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1;
  // Ensure previous speech is stopped before speaking a new one
  speechSynthesis.cancel(); 
  speechSynthesis.speak(utter);
}

// Function to handle speaking the typed text (for mute users)
function speakTypedText() {
  const text = textToSpeakArea.value.trim();
  if (text) {
    speakAndTranscribe(text);
    // Clear the textarea after speaking
    textToSpeakArea.value = ""; 
    // Reset sign recognition output temporarily
    updateUI("Typed: " + text, 1.0);
    // Stop continuous recognition speaking to avoid interruption
    lastDisplayedLabel = text;
  } else {
    alert("Please type a message first.");
  }
}

// Update UI
function updateUI(label, confidence) {
  labelDiv.textContent = label;
  confidenceBar.style.width = Math.round(confidence * 100) + "%";
  confidenceText.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
}

// Predict loop
async function predictLoop() {
  if (!running || !model) {
    rafId = null;
    return;
  }

  // ... (existing prediction logic remains the same)
  // Prepare video frame
  const size = 224;
  let tmp = document.createElement("canvas");
  tmp.width = tmp.height = size;
  let tctx = tmp.getContext("2d");
  // center crop
  const s = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - s) / 2;
  const sy = (video.videoHeight - s) / 2;
  tctx.drawImage(video, sx, sy, s, s, 0, 0, size, size);

  let tensor = tf.browser.fromPixels(tmp).resizeNearestNeighbor([size, size]).toFloat().div(255.0).expandDims(0);
  let pred = model.predict(tensor);
  let data = await pred.data();
  let maxIdx = data.indexOf(Math.max(...data));
  let confidence = data[maxIdx];
  let detectedLabel = labels[maxIdx] || "—";

  // Check if confidence is high enough before announcing/transcribing
  const CONFIDENCE_THRESHOLD = 0.8; // Set a sensible threshold
  if (confidence > CONFIDENCE_THRESHOLD) {
      if (detectedLabel !== lastDisplayedLabel) {
          updateUI(detectedLabel, confidence);
          speakAndTranscribe(detectedLabel);
          lastDisplayedLabel = detectedLabel; // Update last displayed label after successful detection and speech
      }
  } else {
      // Show low confidence or unrecognized state but don't speak
      updateUI(`Unrecognized (${detectedLabel})`, confidence);
  }

  tf.dispose([tensor, pred]);
  rafId = requestAnimationFrame(predictLoop);
}

// --- Button Events ---

// Start button event
startBtn.addEventListener("click", async () => {
  if (!video.srcObject) await startCamera();
  if (video.srcObject) { // Only start recognition if camera is active
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    // Reset UI
    updateUI("Analyzing...", 0);
    spokenTranscription.textContent = "Listening...";
    predictLoop();
  }
});

// Stop button event
stopBtn.addEventListener("click", () => {
  running = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (rafId) cancelAnimationFrame(rafId);
  // Clear status on stop
  updateUI("Recognition Paused", 0);
  spokenTranscription.textContent = "Paused.";
});

// Load model button event
loadModelBtn.addEventListener("click", loadModel);

// Speak typed text button event
speakTextBtn.addEventListener("click", speakTypedText);

// Initial state setup (Disable start until model is loaded)
startBtn.disabled = true;