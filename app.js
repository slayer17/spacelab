(() => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const startBtn = document.getElementById("startBtn");
  const captureBtn = document.getElementById("captureBtn");
  const result = document.getElementById("result");

  let currentStream = null;

  async function startCamera() {
    try {
      currentStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } }
      });
      video.srcObject = currentStream;
      await video.play();
    } catch (err) {
      console.error("Erreur caméra :", err);
      result.textContent = "Erreur caméra : " + err.message;
    }
  }

  function takePhoto() {
    if (!currentStream) {
      result.textContent = "La caméra n'est pas démarrée.";
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    sendToPython();
  }

  async function sendToPython() {
    canvas.toBlob(async (blob) => {
      if (!blob) {
        result.textContent = "Impossible de créer l'image.";
        return;
      }

      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      result.textContent = "Envoi à Railway...";

      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        console.log("Réponse Python :", data);

        drawResults(data.rects || []);
      } catch (err) {
        console.error("Erreur Railway :", err);
        result.textContent = "Le serveur Python ne répond pas.";
      }
    }, "image/jpeg", 0.9);
  }

  function drawResults(rects) {
    if (!Array.isArray(rects) || rects.length === 0) {
      result.textContent = "Aucune zone détectée par Python.";
      return;
    }

    ctx.lineWidth = 3;
    ctx.font = "20px Arial";

    const lines = [];

    rects.forEach((r, index) => {
      if (r.type === "STATION") {
        ctx.strokeStyle = "red";
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        lines.push(`STATION ${index + 1} → x=${r.x}, y=${r.y}, w=${r.w}, h=${r.h}`);
      } else {
        ctx.strokeStyle = "yellow";
        ctx.strokeRect(r.x, r.y, r.w, r.h);

        if (r.name) {
          ctx.fillStyle = "lime";
          ctx.fillText(r.name, r.x, Math.max(20, r.y - 8));
          lines.push(`${r.name} | score=${r.score}`);
        } else {
          lines.push(`CARTE ? | score=${r.score}`);
        }
      }
    });

    result.textContent = lines.join("\n");
  }

  if (startBtn) {
    startBtn.addEventListener("click", startCamera);
  }

  if (captureBtn) {
    captureBtn.addEventListener("click", takePhoto);
  }
})();
