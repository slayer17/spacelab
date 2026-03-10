/* =====================================================
   STATION MATCHER
===================================================== */

window.stationSignatures = [
{
  "id": "STATION_gauche",
   "globalSignature": "1111100000001111111101010001111111110110110011111111001111100111110001101101100110010011110010100001001111111010000100111111101000001011111100100001101111100010000110111111101011001011110110011100001111000011111010111101011111100011110001111110010111101111",
  "centerSignature": "1111111111111111111111111111111111111111000000001111111000110000000000000011111110000000001111110000000001111111001110000111111100000000011111100000000001111110000000000111111000000000011111100000000001111110111111000000001011111100000000001111111111110000",
  "globalColorSignature": {
    "r": 167,
    "g": 164,
    "b": 156
  },
  "crop": {
    "x": 0,
    "y": 0,
    "width": 214,
    "height": 318
  }
},
  {
    "id": "STATION_milieu",
     "globalSignature": "1111100000011111111100001000111111110100001011111111000000001111110000111111101110011101110011101001101111011010110100111111000011101111111000001001101111110010100110111111101010010011111110101101101111000011111000111100011111111011111001111111000000010111",
  "centerSignature": "1111111111111111111111111111111011111111110000001111110000011101110011100111111110000110011000001000011001000000100001100100000010000111011110001000011001110000100001100111111110000110011111111000011001111011100001100000000111111000001111111111111100000000",
  "globalColorSignature": {
    "r": 151,
    "g": 147,
    "b": 140
  },
  "crop": {
    "x": 0,
    "y": 0,
    "width": 220,
    "height": 314
  }
},
{
  "id": "STATION_droite",
   "globalSignature": "1111000000001111111100000110111111110010111011111110001111000111100111100110001100010011110010000001101111001000000110111101100000101011110110000000111111011000000110111101100010011011110100111100001111000011111110111101011111110011111001111111011111110111",
  "centerSignature": "1111111111111100111111111111000011111110000000001100100011011111000110111111111100011011110111111001101000011111100100000001111100010100111100100001000100000000000110000000000000011101111111100001110000111111110001100000111111111100000000001111111111100000",
  "globalColorSignature": {
    "r": 152,
    "g": 146,
    "b": 136
  },
  "crop": {
    "x": 0,
    "y": 0,
    "width": 207,
    "height": 308
  }
}
];

function hammingDistanceStation(a, b) {
  if (!a || !b) return 9999;
  if (a.length !== b.length) return 9999;

  let dist = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) dist++;
  }
  return dist;
}

function rgbDistance(c1, c2) {
  if (!c1 || !c2) return 9999;

  return Math.sqrt(
    Math.pow(c1.r - c2.r, 2) +
    Math.pow(c1.g - c2.g, 2) +
    Math.pow(c1.b - c2.b, 2)
  );
}

function findBestStationMatch(signature) {
  if (!window.stationSignatures || window.stationSignatures.length === 0) {
    return null;
  }

  let bestStation = null;
  let bestDistance = 999999;

  window.stationSignatures.forEach(station => {
    const dGlobal = hammingDistanceStation(
      signature.global,
      station.globalSignature
    );

    let dCenter = 0;
    if (signature.stationCenter && station.centerSignature) {
      dCenter = hammingDistanceStation(
        signature.stationCenter,
        station.centerSignature
      );
    }

    let dColor = 0;
    if (signature.rgbColor && station.globalColorSignature) {
      dColor = rgbDistance(
        signature.rgbColor,
        station.globalColorSignature
      );
    }

    const total =
      dGlobal * 0.60 +
      dCenter * 0.30 +
      dColor * 1.20;

    if (total < bestDistance) {
      bestDistance = total;
      bestStation = station;
    }
  });

  if (!bestStation) return null;

  return {
    station: bestStation,
    distance: bestDistance
  };
}