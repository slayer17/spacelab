function detectColor(imageData) {

  const data = imageData.data;

  let hueTotal = 0;
  let count = 0;

  for (let i = 0; i < data.length; i += 4) {

    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;

    // Ignorer pixels peu saturés (gris/blanc/noir)
    if (delta < 30) continue;

    let hue;

    if (max === r) {
      hue = 60 * (((g - b) / delta) % 6);
    } else if (max === g) {
      hue = 60 * ((b - r) / delta + 2);
    } else {
      hue = 60 * ((r - g) / delta + 4);
    }

    if (hue < 0) hue += 360;

    hueTotal += hue;
    count++;
  }

  if (count === 0) return "ROUGE";

  const avgHue = hueTotal / count;

  // ROUGE
  if (avgHue >= 340 || avgHue <= 20) return "ROUGE";

  // JAUNE
  if (avgHue >= 30 && avgHue <= 70) return "JAUNE";

  // VERT
  if (avgHue >= 80 && avgHue <= 160) return "VERT";

  // BLEU
  if (avgHue >= 190 && avgHue <= 260) return "BLEU";

  return "ROUGE";
}

function detectColorFullROI(imageData) {

  const data = imageData.data;

  let hueTotal = 0;
  let count = 0;

  for (let i = 0; i < data.length; i += 4) {

    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;

    // ignorer pixels peu saturés
    if (delta < 30) continue;

    let hue;

    if (max === r) {
      hue = 60 * (((g - b) / delta) % 6);
    } else if (max === g) {
      hue = 60 * ((b - r) / delta + 2);
    } else {
      hue = 60 * ((r - g) / delta + 4);
    }

    if (hue < 0) hue += 360;

    hueTotal += hue;
    count++;
  }

  if (count === 0) return "INCONNU";

  const avgHue = hueTotal / count;

  console.log("Carte 0 Hue moyen ROI complet :", avgHue);

  if (avgHue >= 150 && avgHue <= 260) return "BLEU";
  if (avgHue >= 80 && avgHue <= 160) return "VERT";
  if (avgHue >= 30 && avgHue <= 70) return "JAUNE";
  if (avgHue >= 340 || avgHue <= 20) return "ROUGE";

  return "INCONNU";
}