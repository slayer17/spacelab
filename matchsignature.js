function toArray(v) {
    if (Array.isArray(v)) return v;
    if (v == null) return [];
    if (typeof v === "number") return [v];
    if (typeof v === "object") return Object.values(v).flatMap(toArray);
    return [];
}

function euclideanDistance(a, b) {
    const aa = toArray(a);
    const bb = toArray(b);
    const n = Math.min(aa.length, bb.length);

    if (n === 0) return Infinity;

    let sum = 0;
    for (let i = 0; i < n; i++) {
        const da = Number(aa[i]) || 0;
        const db = Number(bb[i]) || 0;
        const d = da - db;
        sum += d * d;
    }
    return Math.sqrt(sum / n);
}

function similarityScore(a, b) {
    const dist = euclideanDistance(a, b);
    if (!isFinite(dist)) return 0;
    return 1 / (1 + dist);
}

function getScanPart(signature, part) {
    if (!signature) return null;
    if (signature.scan && signature.scan[part] != null) return signature.scan[part];
    return signature[part] != null ? signature[part] : null;
}

function enrichCandidate(querySig, card) {
    const cardSig = card.signature || card;

    const colorScore = similarityScore(
        getScanPart(querySig, "color"),
        getScanPart(cardSig, "color")
    );

    const symbolScore = similarityScore(
        getScanPart(querySig, "symbol"),
        getScanPart(cardSig, "symbol")
    );

    const bottomScore = similarityScore(
        getScanPart(querySig, "bottom"),
        getScanPart(cardSig, "bottom")
    );

    const globalScore = similarityScore(
        getScanPart(querySig, "global"),
        getScanPart(cardSig, "global")
    );

    return {
        card,
        colorScore,
        symbolScore,
        bottomScore,
        globalScore
    };
}

function keepBestBy(candidates, key, options = {}) {
    const {
        keepTop = 8,
        ratio = 0.92,
        minKeep = 2
    } = options;

    if (!candidates.length) return [];

    const sorted = [...candidates].sort((a, b) => b[key] - a[key]);
    const best = sorted[0][key];

    let kept = sorted.filter(c => c[key] >= best * ratio);

    if (kept.length < minKeep) {
        kept = sorted.slice(0, Math.min(keepTop, sorted.length));
    } else {
        kept = kept.slice(0, keepTop);
    }

    return kept;
}

function finalTieBreak(candidates) {
    if (!candidates.length) return null;

    const scored = candidates.map(c => ({
        ...c,
        finalScore:
            (c.colorScore * 0.15) +
            (c.symbolScore * 0.35) +
            (c.bottomScore * 0.40) +
            (c.globalScore * 0.10)
    }));

    scored.sort((a, b) => b.finalScore - a.finalScore);
    return scored[0];
}

/**
 * Match hiérarchique :
 * 1) color
 * 2) symbol
 * 3) bottom
 * 4) global en secours
 *
 * @param {Object} querySig signature calculée sur la carte scannée
 * @param {Array} cardsDb base des cartes/signatures
 * @returns {Object|null}
 */
function matchSignature(querySig, cardsDb) {
    if (!querySig || !cardsDb || !cardsDb.length) return null;

    // 1) enrichir tous les candidats
    let candidates = cardsDb.map(card => enrichCandidate(querySig, card));

    // 2) filtre couleur
    let stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });

    // 3) filtre symbole
    let stepSymbol = keepBestBy(stepColor, "symbolScore", {
        keepTop: 6,
        ratio: 0.90,
        minKeep: 2
    });

    // 4) filtre bottom
    let stepBottom = keepBestBy(stepSymbol, "bottomScore", {
        keepTop: 3,
        ratio: 0.92,
        minKeep: 1
    });

    // 5) choix final
    let best = finalTieBreak(stepBottom);

    // Fallback si on n’a rien de convaincant
    if (!best) {
        const fallback = [...candidates]
            .map(c => ({
                ...c,
                finalScore:
                    (c.colorScore * 0.25) +
                    (c.symbolScore * 0.25) +
                    (c.bottomScore * 0.25) +
                    (c.globalScore * 0.25)
            }))
            .sort((a, b) => b.finalScore - a.finalScore);

        best = fallback[0] || null;
    }

    if (!best) return null;

    return {
        card: best.card,
        score: best.finalScore ?? (
            (best.colorScore * 0.15) +
            (best.symbolScore * 0.35) +
            (best.bottomScore * 0.40) +
            (best.globalScore * 0.10)
        ),
        debug: {
            colorScore: best.colorScore,
            symbolScore: best.symbolScore,
            bottomScore: best.bottomScore,
            globalScore: best.globalScore,
            colorCandidates: stepColor.map(c => ({
                id: c.card.id,
                name: c.card.name,
                score: c.colorScore
            })),
            symbolCandidates: stepSymbol.map(c => ({
                id: c.card.id,
                name: c.card.name,
                score: c.symbolScore
            })),
            bottomCandidates: stepBottom.map(c => ({
                id: c.card.id,
                name: c.card.name,
                score: c.bottomScore
            }))
        }
    };
}