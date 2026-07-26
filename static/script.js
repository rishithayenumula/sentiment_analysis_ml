const reviewInput = document.getElementById("reviewInput");
const submitBtn = document.getElementById("submitBtn");
const caseNumber = document.getElementById("caseNumber");

const reportEmpty = document.getElementById("reportEmpty");
const reportContent = document.getElementById("reportContent");
const reportError = document.getElementById("reportError");
const errorText = document.getElementById("errorText");

const stampEl = document.getElementById("stampEl");
const stampText = document.getElementById("stampText");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeNeedle = document.getElementById("gaugeNeedle");
const confidenceValue = document.getElementById("confidenceValue");
const posFill = document.getElementById("posFill");
const negFill = document.getElementById("negFill");
const posPct = document.getElementById("posPct");
const negPct = document.getElementById("negPct");
const filedTimestamp = document.getElementById("filedTimestamp");
const modelName = document.getElementById("modelName");

let caseCounter = Math.floor(Math.random() * 900) + 100;
caseNumber.textContent = `${caseCounter}-01`;

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    reviewInput.value = chip.textContent;
    reviewInput.focus();
  });
});

const GAUGE_ARC_LENGTH = 267; // matches stroke-dasharray in CSS

async function submitText() {
  const text = reviewInput.value.trim();
  if (!text) {
    reviewInput.focus();
    return;
  }

  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-label").textContent = "Inspecting...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "The inspection could not be completed.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Could not reach the inspection service. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector(".btn-label").textContent = "Submit for Inspection";
  }
}

function showError(message) {
  reportEmpty.hidden = true;
  reportContent.hidden = true;
  reportError.hidden = false;
  errorText.textContent = message;
}

function renderResult(data) {
  reportEmpty.hidden = true;
  reportError.hidden = true;
  reportContent.hidden = false;

  const sentiment = data.sentiment; // "positive" | "negative"
  const confidence = data.confidence ?? 0;
  const breakdown = data.breakdown || {};

  // Stamp
  stampEl.classList.remove("stamped", "positive", "negative");
  void stampEl.offsetWidth; // restart animation
  stampEl.classList.add(sentiment === "positive" ? "positive" : "negative");
  stampText.textContent = sentiment.toUpperCase();
  requestAnimationFrame(() => stampEl.classList.add("stamped"));

  // Gauge
  const pct = Math.round(confidence * 100);
  const offset = GAUGE_ARC_LENGTH * (1 - confidence);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = sentiment === "positive" ? "#3c6e47" : "#a5342a";
  // needle sweeps from -90deg (left) to +90deg (right) across confidence 0..1
  const angle = -90 + confidence * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  confidenceValue.textContent = `${pct}%`;

  // Breakdown bars
  const posP = breakdown.positive ?? (sentiment === "positive" ? confidence : 1 - confidence);
  const negP = breakdown.negative ?? (sentiment === "negative" ? confidence : 1 - confidence);
  posFill.style.width = `${Math.round(posP * 100)}%`;
  negFill.style.width = `${Math.round(negP * 100)}%`;
  posPct.textContent = `${Math.round(posP * 100)}%`;
  negPct.textContent = `${Math.round(negP * 100)}%`;

  // Metadata
  filedTimestamp.textContent = new Date().toLocaleString();
  modelName.textContent = "TF-IDF + Linear Classifier";

  caseCounter += 1;
  caseNumber.textContent = `${caseCounter}-01`;
}

submitBtn.addEventListener("click", submitText);
reviewInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    submitText();
  }
});
