/* =====================================================
   CARD MATCHER VERSION STABLE MINIMALE
===================================================== */

function hammingDistance(a, b) {
  if (!a || !b) return 9999;
  if (a.length !== b.length) return 9999;

  let dist = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) dist++;
  }
  return dist;
}

function colorDistance(c1, c2) {
  if (!c1 || !c2) return 0;

  return Math.sqrt(
    Math.pow(c1.h - c2.h, 2) +
    Math.pow(c1.s - c2.s, 2) +
    Math.pow(c1.l - c2.l, 2)
  );
}

function findBestMatch(signature, blobColor, usedCards) {

  if (!window.cards) return null;

  let bestCard = null;
  let bestDistance = 9999;

  let candidates = window.cards.filter(
    c => c.couleur === blobColor
  );

  if (candidates.length === 0) {
    candidates = window.cards;
  }

  candidates.forEach(card => {

    if (usedCards && usedCards.has(card.id)) return;

    if (!card.signature) return;

    const refSig =
      window.MODE === "BOARD"
        ? card.signature.board
        : card.signature.scan;

    if (!refSig) return;

    const dGlobal = hammingDistance(
      signature.global,
      refSig.globalSignature
    );

    const dSymbole = hammingDistance(
      signature.symbole,
      refSig.symboleSignature
    );
const dPoints = hammingDistance(
  signature.points,
  refSig.pointsSignature
);

    let dColor = 0;
    if (signature.color && refSig.globalColorSignature) {
      dColor = colorDistance(
        signature.color,
        refSig.globalColorSignature
      );
    }

  const total =
  dSymbole * 0.55 +
  dPoints * 0.30 +
  dGlobal * 0.15 +
  dColor * 40;

    if (total < bestDistance) {
      bestDistance = total;
      bestCard = card;
    }

  });

  if (!bestCard) return null;

  return {
    card: bestCard,
    distance: bestDistance
  };
}