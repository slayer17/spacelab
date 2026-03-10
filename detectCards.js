// =====================================================
// 1️⃣ Détection couleur du fond detectcard.js
// =====================================================

function detectBackgroundColor(ctx, canvas) {

  const bandWidth = 30;

  let r = 0, g = 0, b = 0;
  let count = 0;

  const { width, height } = canvas;

  const left = ctx.getImageData(0, 0, bandWidth, height).data;
  const right = ctx.getImageData(width - bandWidth, 0, bandWidth, height).data;

  for (let i = 0; i < left.length; i += 4) {
    r += left[i];
    g += left[i + 1];
    b += left[i + 2];
    count++;
  }

  for (let i = 0; i < right.length; i += 4) {
    r += right[i];
    g += right[i + 1];
    b += right[i + 2];
    count++;
  }

  return {
    r: Math.round(r / count),
    g: Math.round(g / count),
    b: Math.round(b / count)
  };
}


// =====================================================
// 2️⃣ Détection couleur carte
// =====================================================

function detectCardColor(ctx, blob) {

  const cropX = Math.floor(blob.x + blob.width * 0.2);
  const cropY = Math.floor(blob.y + blob.height * 0.2);
  const cropW = Math.floor(blob.width * 0.6);
  const cropH = Math.floor(blob.height * 0.6);

  const imageData = ctx.getImageData(cropX, cropY, cropW, cropH);
  const data = imageData.data;

  let r = 0, g = 0, b = 0, count = 0;

  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    count++;
  }

  r /= count;
  g /= count;
  b /= count;

  const max = Math.max(r, g, b);

  if (g === max) return "VERT";
  if (r === max) return "ROUGE";
  if (b === max) return "BLEU";

  return "UNKNOWN";
}


// =====================================================
// 3️⃣ Fusion blobs proches
// =====================================================

function mergeNearbyBlobs(blobs) {

  const used = new Array(blobs.length).fill(false);
  const merged = [];

  for (let i = 0; i < blobs.length; i++) {

    if (used[i]) continue;

    const a = blobs[i];

    let minX = a.x;
    let minY = a.y;
    let maxX = a.x + a.width;
    let maxY = a.y + a.height;

    for (let j = i + 1; j < blobs.length; j++) {

      if (used[j]) continue;

      const b = blobs[j];

      const dx = Math.abs(a.x - b.x);
      const dy = Math.abs(a.y - b.y);

      // fusion seulement si très proche
      if (dx < 60 && dy < 80) {
        minX = Math.min(minX, b.x);
        minY = Math.min(minY, b.y);
        maxX = Math.max(maxX, b.x + b.width);
        maxY = Math.max(maxY, b.y + b.height);
        used[j] = true;
      }
    }

    merged.push({
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY
    });
  }

  return merged;
}

function median(values) {
  if (!values || values.length === 0) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  return sorted[mid];
}

function pickLargestBlob(blobs) {
  if (!blobs || blobs.length === 0) return null;

  return [...blobs].sort(
    (a, b) => (b.width * b.height) - (a.width * a.height)
  )[0];
}
// =====================================================
// 4️⃣ Score d’un triplet de stations
// =====================================================

function getStationTripletScore(a, b, c) {

  const triplet = [a, b, c].sort((p1, p2) => p1.x - p2.x);
  const [left, center, right] = triplet;

  const leftCenterX = left.x + left.width / 2;
  const centerCenterX = center.x + center.width / 2;
  const rightCenterX = right.x + right.width / 2;

  const leftCenterY = left.y + left.height / 2;
  const centerCenterY = center.y + center.height / 2;
  const rightCenterY = right.y + right.height / 2;

  const avgWidth = (left.width + center.width + right.width) / 3;
  const avgHeight = (left.height + center.height + right.height) / 3;
  const avgY = (leftCenterY + centerCenterY + rightCenterY) / 3;

  const gap1 = centerCenterX - leftCenterX;
  const gap2 = rightCenterX - centerCenterX;

  if (gap1 <= 0 || gap2 <= 0) return Infinity;

  const widthPenalty =
    Math.abs(left.width - avgWidth) +
    Math.abs(center.width - avgWidth) +
    Math.abs(right.width - avgWidth);

  const heightPenalty =
    Math.abs(left.height - avgHeight) +
    Math.abs(center.height - avgHeight) +
    Math.abs(right.height - avgHeight);

  const yPenalty =
    Math.abs(leftCenterY - avgY) +
    Math.abs(centerCenterY - avgY) +
    Math.abs(rightCenterY - avgY);

  const gapPenalty = Math.abs(gap1 - gap2);

  return widthPenalty + heightPenalty + yPenalty * 2 + gapPenalty * 2;
}


// =====================================================
// 5️⃣ Sélection intelligente du triplet de stations
// =====================================================

function selectBestStationTriplet(candidates) {

  if (candidates.length < 3) return [];

  let bestTriplet = [];
  let bestScore = Infinity;

  for (let i = 0; i < candidates.length - 2; i++) {
    for (let j = i + 1; j < candidates.length - 1; j++) {
      for (let k = j + 1; k < candidates.length; k++) {

        const a = candidates[i];
        const b = candidates[j];
        const c = candidates[k];

        const score = getStationTripletScore(a, b, c);

        if (score < bestScore) {
          bestScore = score;
          bestTriplet = [a, b, c].sort((p1, p2) => p1.x - p2.x);
        }
      }
    }
  }

  console.log("Meilleur score stations :", bestScore);

  return bestTriplet;
}


// =====================================================
// 6️⃣ Détection plateau
// =====================================================

function detectCards(ctx) {

  const canvas = ctx.canvas;
  const width = canvas.width;
  const height = canvas.height;

  console.log("----- DETECTION BOARD -----");

  const image = ctx.getImageData(0,0,width,height);
  const data = image.data;

  const bg = detectBackgroundColor(ctx, canvas);

  const mask = new Uint8Array(width * height);

  const threshold = 45;

  for(let i=0;i<data.length;i+=4){

    const r = data[i];
    const g = data[i+1];
    const b = data[i+2];

    const d = Math.sqrt(
      (r-bg.r)*(r-bg.r) +
      (g-bg.g)*(g-bg.g) +
      (b-bg.b)*(b-bg.b)
    );

    mask[i/4] = d > threshold ? 1 : 0;
  }

  const visited = new Uint8Array(width*height);
  const blobs = [];

  function idx(x,y){
    return y*width + x;
  }

  function floodFill(x,y){

    const stack = [[x,y]];

    let minX=x,maxX=x,minY=y,maxY=y;

    visited[idx(x,y)] = 1;

    while(stack.length){

      const [cx,cy] = stack.pop();

      minX=Math.min(minX,cx);
      maxX=Math.max(maxX,cx);
      minY=Math.min(minY,cy);
      maxY=Math.max(maxY,cy);

      const n=[
        [cx+1,cy],
        [cx-1,cy],
        [cx,cy+1],
        [cx,cy-1]
      ];

      for(const [nx,ny] of n){

        if(
          nx>=0 && nx<width &&
          ny>=0 && ny<height &&
          !visited[idx(nx,ny)] &&
          mask[idx(nx,ny)]
        ){
          visited[idx(nx,ny)]=1;
          stack.push([nx,ny]);
        }

      }

    }

    return {
      x:minX,
      y:minY,
      width:maxX-minX,
      height:maxY-minY
    };

  }

for(let y=0;y<height;y++){
  for(let x=0;x<width;x++){

    if(!visited[idx(x,y)] && mask[idx(x,y)]){

      const blob = floodFill(x,y);

      // filtre anti-bruit
      if(blob.width < width * 0.02) continue;
      if(blob.height < height * 0.05) continue;

      blobs.push(blob);

    }

  }
}
  console.log("blobs:",blobs.length);

  const merged = mergeNearbyBlobs(blobs);

  const vertical = merged.filter(b=>{
    const r=b.height/b.width;
    return r>1.1 && r<2.5;
  });

  const refW = median(vertical.map(b=>b.width));
  const refH = median(vertical.map(b=>b.height));

  console.log("ref card:",refW,refH);

// ==============================
// Détection stations par colonne
// ==============================

// const stations = [];

// const columnWidth = width / 3;

// for(let i = 0; i < 3; i++){

  // const colMin = i * columnWidth;
  // const colMax = (i + 1) * columnWidth;

  // const candidates = vertical.filter(b => {

    // const cx = b.x + b.width / 2;

    // return cx > colMin && cx < colMax;

  // });

  // if(candidates.length){

    // const station = candidates.sort(
      // (a,b) => (b.width*b.height) - (a.width*a.height)
    // )[0];

    // stations.push(station);

  // }

// }
// ==============================
// Détection stations (simple et stable)
// ==============================

const stations = vertical
  .sort((a,b) => b.height - a.height)
  .slice(0,3)
  .sort((a,b) => a.x - b.x);
  stations.forEach(s=>s.type="STATION");

  const cards = vertical.filter(b=>{

    if(stations.includes(b)) return false;

    if(b.width < refW*0.7) return false;
    if(b.width > refW*1.2) return false;

    if(b.height < refH*0.7) return false;
    if(b.height > refH*1.2) return false;

    b.type="CARTE";
    b.couleur = detectCardColor(ctx,b);

    return true;

  });

  const objects=[...stations,...cards];

  const centers = stations.map(s=>s.x+s.width/2);

  function getColumn(o){

    const cx=o.x+o.width/2;

    let best=0;
    let dist=Infinity;

    centers.forEach((c,i)=>{
      const d=Math.abs(cx-c);
      if(d<dist){
        dist=d;
        best=i;
      }
    });

    return best;

  }

  objects.forEach(o=>{
    o.column=getColumn(o);
  });

  console.log("stations:",stations.length);
  console.log("cards:",cards.length);

  return objects;
}