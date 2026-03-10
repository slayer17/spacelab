(() => {
/* =====================================================
   1 - CONFIGURATION ET DOM
===================================================== */
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const loadBtn = document.getElementById("loadBtn");
const fileInput = document.getElementById("file");

let currentStream = null;

/* =====================================================
   2 - CAMERA ET CAPTURE
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

// Cette fonction prend la photo et l'envoie à Python
function takePhoto() {
  if (!currentStream) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  // On transforme l'image du canvas en un fichier (blob) pour l'envoyer
  canvas.toBlob((blob) => {
    getDetectionsFromServer(blob);
  }, 'image/jpeg', 0.9);
}

/* =====================================================
   3 - CONNEXION AVEC LE SERVEUR PYTHON (RAILWAY)
===================================================== */

async function getDetectionsFromServer(blob) {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');

    console.log("Envoi de l'image au serveur Python...");

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Erreur serveur Railway");

        const data = await response.json();
        console.log("Rectangles reçus de Python:", data.rects);
        
        // On dessine les rectangles trouvés par Python
        processPythonResults(data.rects);

    } catch (err) {
        console.error("Erreur lors de l'envoi au serveur:", err);
        alert("Le serveur Python ne répond pas. Vérifie ton déploiement Railway.");
    }
}

function processPythonResults(rects) {
    // usedCards permet d'éviter de détecter deux fois la même carte
    let usedCards = new Set();
    let resume = "";

    rects.forEach((rect, index) => {
        // 1. On dessine le rectangle de base trouvé par Python
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 5;
        ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

        // 2. EXTRACTION DE LA ZONE (Le "Cerveau")
        // On récupère l'image précise à l'intérieur du rectangle
        const cardData = ctx.getImageData(rect.x, rect.y, rect.w, rect.h);
        
        // 3. RECONNAISSANCE (Appel à tes autres fichiers JS)
        // Ici on utilise ta logique existante de detectColor et cardmatcher
        const cardColor = detectColor(cardData); // Vient de detectColor.js
        
        // On crée une signature rapide pour la zone (simplifié)
        // Note : Si tu avais une fonction spécifique pour générer la signature, appelle-la ici
        const signature = { global: "0000...", symbole: "0000...", points: "0000..." }; 

        // Appel de ton moteur de match
        const match = findBestMatch(signature, cardColor, usedCards);

        if (match && match.card) {
            usedCards.add(match.card.id);
            
            // On affiche le nom de la carte sur le canvas
            ctx.fillStyle = "yellow";
            ctx.font = "bold 20px Arial";
            ctx.fillText(match.card.id, rect.x + 10, rect.y + 25);
            
            resume += `<div>Carte ${index + 1}: <b>${match.card.id}</b></div>`;
        } else {
            ctx.fillStyle = "orange";
            ctx.fillText("Inconnu", rect.x + 10, rect.y + 25);
        }
    });

    // On affiche le résultat dans ta zone de texte
    const resultDiv = document.getElementById("result");
    if (resultDiv) resultDiv.innerHTML = resume;
}

/* =====================================================
   4 - ÉVÉNEMENTS (CONNEXION BOUTONS)
===================================================== */
const startBtn = document.getElementById("startBtn");
// On relie le bouton "Démarrer caméra"
if(startBtn) startBtn.addEventListener("click", startCamera);

// On relie le bouton "Capturer" à la fonction takePhoto
if(captureBtn) captureBtn.addEventListener("click", takePhoto);

// Gestion du chargement de fichier local
if(loadBtn) loadBtn.addEventListener("click", () => fileInput.click());
if(fileInput) fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => getDetectionsFromServer(blob), 'image/jpeg');
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

})(); 
