(() => {

/* =====================================================
   1 - DOM
===================================================== */
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const loadBtn = document.getElementById("loadBtn");
const fileInput = document.getElementById("file");
const modeBtn = document.getElementById("modeBtn");
const cardName = document.getElementById("cardName");

let currentStream = null;

/* =====================================================
   CAMERA
===================================================== */
async function startCamera() {
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }
    });
    video.srcObject = currentStream;
    await video.play();
  } catch (err) {
    console.error("Erreur caméra :", err);
  }
}

// MODIFICATION : On envoie à Python au lieu d'analyser en local direct
function takePhoto() {
  if (!currentStream) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  // Si on est en mode CARDS_ONLY, on analyse direct (pas besoin de Python pour 1 seule carte)
  if (window.MODE === "CARDS_ONLY") {
      analyzeCanvas();
  } else {
      // Si on est en mode BOARD, on demande à Python de trouver les zones
      canvas.toBlob((blob) => {
        getDetectionsFromServer(blob);
      }, 'image/jpeg', 0.9);
  }
}

const startBtn = document.getElementById("startBtn");
if(startBtn) startBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", takePhoto);

/* =====================================================
   CONFIG
===================================================== */
// On initialise la variable sur l'objet window pour qu'elle soit "Générale"
window.MODE = "BOARD"; 
const HASH_SIZE = 16;

/* =====================================================
   SWITCH MODE
===================================================== */
modeBtn.addEventListener("click", () => {
  // On change la valeur
  window.MODE = (window.MODE === "BOARD") ? "CARDS_ONLY" : "BOARD";
  
  // On met à jour le texte du bouton pour VÉRIFIER que ça marche
  modeBtn.textContent = "Mode: " + window.MODE;
  
  console.log("Mode changé en direct →", window.MODE);
});

/* =====================================================
   LOAD IMAGE
===================================================== */
loadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    // Même logique : Python si Board, direct si Cards_only
    if (window.MODE === "BOARD") {
        canvas.toBlob((blob) => getDetectionsFromServer(blob), 'image/jpeg');
    } else {
        analyzeCanvas();
    }
  };
  img.src = URL.createObjectURL(file);
});

/* =====================================================
   CONNEXION RAILWAY (NOUVEAU)
===================================================== */
async function getDetectionsFromServer(blob) {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');
    console.log("Envoi à Railway pour détection des zones...");
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        // data.rects contient les x, y, w, h trouvés par Python
        processPythonResults(data.rects);
    } catch (err) {
        console.error("Erreur Railway:", err);
        alert("Le serveur Python ne répond pas.");
    }
}

function processPythonResults(rects) {
    // On transforme les rects de Python en format compatible avec ton code
    const objects = rects.map(r => ({
        x: r.x,
        y: r.y,
        width: r.w,
        height: r.h,
        type: (r.w > r.h) ? "STATION" : "CARTE" // Devine le type par la forme
    }));

    // On lance ton analyse habituelle mais avec les zones de Python !
    analyzeCanvas(objects);
}

/* =====================================================
   TES FONCTIONS DE HASH ET COULEUR (CONSERVÉES)
===================================================== */
function computeGlobalColor(imageData) {
  const data = imageData.data;
  let totalH = 0, totalS = 0, totalL = 0, count = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
    const max = Math.max(r,g,b), min = Math.min(r,g,b), delta = max - min;
    let h = 0, s = 0, l = (max + min) / 2;
    if (delta !== 0) {
      s = delta / (1 - Math.abs(2*l - 1));
      switch(max){
        case r: h = ((g-b)/delta) % 6; break;
        case g: h = (b-r)/delta + 2; break;
        case b: h = (r-g)/delta + 4; break;
      }
      h = Math.round(h * 60); if (h < 0) h += 360;
    }
    totalH += h; totalS += s; totalL += l; count++;
  }
  return { h: totalH / count, s: totalS / count, l: totalL / count };
}

function computeAverageRGB(imageData) {
  const data = imageData.data;
  let totalR = 0, totalG = 0, totalB = 0, count = 0;
  for (let i = 0; i < data.length; i += 4) {
    totalR += data[i]; totalG += data[i + 1]; totalB += data[i + 2]; count++;
  }
  return { r: Math.round(totalR / count), g: Math.round(totalG / count), b: Math.round(totalB / count) };
}

function computePerceptualHash(zone) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 200; tempCanvas.height = 300;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, zone.x, zone.y, zone.width, zone.height, 0, 0, 200, 300);
  const fullData = tempCtx.getImageData(0, 0, 200, 300);

  const symbolData = tempCtx.getImageData(10, 45, 60, 60); // Ajusté selon tes ratios
  const pointsData = tempCtx.getImageData(10, 225, 100, 60);
  const stationCenterData = tempCtx.getImageData(10, 40, 45, 210);

  return {
    global: computeHash(fullData, HASH_SIZE),
    symbole: computeHash(symbolData, HASH_SIZE),
    points: computeHash(pointsData, HASH_SIZE),
    stationCenter: computeHash(stationCenterData, HASH_SIZE),
    color: computeGlobalColor(fullData),
    rgbColor: computeAverageRGB(fullData)
  };
}

function computeHash(imageData, size) {
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  tempCanvas.width = size; tempCanvas.height = size;
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = imageData.width; sourceCanvas.height = imageData.height;
  sourceCanvas.getContext("2d").putImageData(imageData, 0, 0);
  tempCtx.drawImage(sourceCanvas, 0, 0, size, size);
  const data = tempCtx.getImageData(0, 0, size, size).data;
  let gray = [];
  for (let i = 0; i < data.length; i += 4) {
    gray.push(data[i] * 0.3 + data[i + 1] * 0.59 + data[i + 2] * 0.11);
  }
  const avg = gray.reduce((a, b) => a + b, 0) / gray.length;
  return gray.map(v => v > avg ? "1" : "0").join("");
}

/* =====================================================
   ANALYSE (VERSION FUSIONNÉE)
===================================================== */
function analyzeCanvas(pythonObjects = null) {
  if (window.MODE === "CARDS_ONLY") {
    const signature = computePerceptualHash({ x: 0, y: 0, width: canvas.width, height: canvas.height });
    const match = findBestMatch(signature, null);
    cardName.textContent = (match && match.card) ? match.card.id + " (" + match.distance.toFixed(1) + ")" : "Aucune correspondance";
    return;
  }

  // MODE BOARD
  // On utilise soit les objets de Python, soit rien
  const filtered = pythonObjects || [];
  if (filtered.length === 0) {
    cardName.textContent = "Aucune zone détectée par Python";
    return;
  }

  // Tri comme avant
  filtered.sort((a, b) => {
    const rowTolerance = 140;
    if (Math.abs(a.y - b.y) > rowTolerance) return a.y - b.y;
    return a.x - b.x;
  });

  const usedCards = new Set();
  let resume = "";

  filtered.forEach((zone, index) => {
    const signature = computePerceptualHash(zone);
    const stationMatch = findBestStationMatch(signature);
    const cardMatch = findBestMatch(signature, zone.couleur, usedCards);

    const isGeometricStation = zone.type === "STATION";
    const isGeometricCard = zone.type === "CARTE";

    // Dessin et Logique identique à ta version
    ctx.lineWidth = 3;
    if (isGeometricStation) {
      ctx.strokeStyle = "red";
      ctx.strokeRect(zone.x, zone.y, zone.width, zone.height);
      resume += `<div>ZONE STATION ${index + 1}</div>`;
    } else {
      if (cardMatch && cardMatch.card && cardMatch.distance < 900) {
        usedCards.add(cardMatch.card.id);
        ctx.strokeStyle = "yellow";
        ctx.strokeRect(zone.x, zone.y, zone.width, zone.height);
        ctx.fillStyle = "lime";
        ctx.fillText(cardMatch.card.id, zone.x + 10, zone.y + 25);
        resume += `<div>${cardMatch.card.id}</div>`;
      } else {
        ctx.strokeStyle = "orange";
        ctx.strokeRect(zone.x, zone.y, zone.width, zone.height);
        resume += `<div>CARTE ? zone ${index + 1}</div>`;
      }
    }
  });
  cardName.innerHTML = resume;
}

})();
