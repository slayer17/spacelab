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

function meanStdScore(a, b) {
    if (!a || !b) return 0;

    if (a.mean == null || b.mean == null) {
        return similarityScore(a, b);
    }

    const dMean = Math.abs(a.mean - b.mean);
    const dStd = Math.abs(a.std - b.std);

    const max = 255;
    const score = 1 - (dMean + dStd) / (2 * max);

    return Math.max(0, score);
}

function patchScore(a, b) {
    if (!a || !b) return 0;

    const vectorA = a.vector || null;
    const vectorB = b.vector || null;

    const vecScore =
        vectorA && vectorB
            ? similarityScore(vectorA, vectorB)
            : 0;

    const statsScore = meanStdScore(a, b);

    if (vectorA && vectorB) {
        return (vecScore * 0.75) + (statsScore * 0.25);
    }

    return statsScore;
}

function getScanPart(signature, part) {
    if (!signature) return null;

    if (signature.scan && signature.scan[part] != null) {
        return signature.scan[part];
    }

    if (signature[part] != null) {
        return signature[part];
    }

    return null;
}

/*
    Cette fonction récupère le chiffre détecté côté scan.
    Important :
    - si le chiffre n'existe pas, on renvoie null
    - si le chiffre existe, on le convertit en nombre
*/
function getReliableDetectedPoints(querySig) {
    const points = getScanPart(querySig, "points");

    if (!points) return null;
    if (points.digit == null) return null;

    const digit = Number(points.digit);

    if (!isFinite(digit)) return null;

    return digit;
}

/*
    Score des points :
    - 1.00 si le chiffre détecté correspond exactement
    - 0.15 si on a 1 point d'écart
    - 0.00 sinon
    - 0.50 neutre si aucun chiffre fiable n'a été détecté
*/
function pointsMatchScore(querySig, card) {
    const detectedPoints = getReliableDetectedPoints(querySig);

    if (detectedPoints == null) {
        return 0.50;
    }

    const cardPoints = Number(card.points);

    if (!isFinite(cardPoints)) {
        return 0;
    }

    if (cardPoints === detectedPoints) {
        return 1.00;
    }

    if (Math.abs(cardPoints - detectedPoints) === 1) {
        return 0.15;
    }

    return 0.00;
}

function enrichCandidate(querySig, card) {
    const cardSig = card.signature || card;

    const qColor = getScanPart(querySig, "color")?.color;
    const cColor = getScanPart(cardSig, "color")?.color;

    let colorScore = 0;

    if (qColor && cColor) {
        const d =
            Math.abs(qColor[0] - cColor[0]) +
            Math.abs(qColor[1] - cColor[1]) +
            Math.abs(qColor[2] - cColor[2]);

        colorScore = 1 / (1 + d / 50);
    }

    const symbolScore = meanStdScore(
        getScanPart(querySig, "symbol"),
        getScanPart(cardSig, "symbol")
    );

    const pointsScore = pointsMatchScore(querySig, card);

    const bottomScore = patchScore(
        getScanPart(querySig, "bottom"),
        getScanPart(cardSig, "bottom")
    );

    const globalScore = patchScore(
        getScanPart(querySig, "global"),
        getScanPart(cardSig, "global")
    );

    return {
        card,
        colorScore,
        symbolScore,
        pointsScore,
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
            (c.symbolScore * 0.45) +
            (c.pointsScore * 0.20) +
            (c.bottomScore * 0.15) +
            (c.globalScore * 0.05)
    }));

    scored.sort((a, b) => b.finalScore - a.finalScore);

    return scored[0];
}

function matchSignature(querySig, cardsDb) {
    if (!querySig || !cardsDb || !cardsDb.length) {
        return {
            card: cardsDb ? cardsDb[0] : null,
            score: 0
        };
    }

    let candidates = cardsDb.map(card => enrichCandidate(querySig, card));

    // -----------------
    // 1) COULEUR
    // -----------------
    let stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });

    // -----------------
    // 2) SYMBOLE
    // -----------------
    const detectedSymbol =
        getScanPart(querySig, "symbol")?.name || null;

    const detectedSymbolScore =
        Number(getScanPart(querySig, "symbol")?.score || 0);

    let stepSymbol = stepColor;

    if (detectedSymbol && detectedSymbolScore >= 0.50) {
        const filtered = stepColor.filter(c => {
            if (!c.card.symbol) return false;

            return (
                c.card.symbol.toUpperCase().trim() ===
                detectedSymbol.toUpperCase().trim()
            );
        });

        if (filtered.length > 0) {
            stepSymbol = filtered;
            console.log("SYMBOL EXACT =", detectedSymbol);
        } else {
            console.log("SYMBOL FILTER FAILED -> keep color");
        }
    } else {
        console.log("SYMBOL UNKNOWN");
    }

    // -----------------
    // 3) POINTS
    // -----------------
    const detectedPoints = getReliableDetectedPoints(querySig);

    let stepPoints = stepSymbol;

    if (detectedPoints != null) {
        const exactPoints = stepSymbol.filter(c => {
            return Number(c.card.points) === detectedPoints;
        });

        if (exactPoints.length > 0) {
            stepPoints = exactPoints;
            console.log("POINTS EXACT =", detectedPoints);
        } else {
            console.log("POINTS FILTER FAILED -> keep symbol");
        }
    } else {
        console.log("POINTS UNKNOWN");
    }

    // -----------------
    // 4) BAS DE CARTE
    // -----------------
    let stepBottom = keepBestBy(stepPoints, "bottomScore", {
        keepTop: 4,
        ratio: 0.97,
        minKeep: 1
    });

    // -----------------
    // 5) SCORE GLOBAL
    // -----------------
    let stepGlobal = keepBestBy(stepBottom, "globalScore", {
        keepTop: 2,
        ratio: 0.97,
        minKeep: 1
    });

    let best = finalTieBreak(stepGlobal);

    // -----------------
    // fallback
    // -----------------
    if (!best) {
        const fallback = [...candidates]
            .map(c => ({
                ...c,
                finalScore:
                    (c.colorScore * 0.15) +
                    (c.symbolScore * 0.45) +
                    (c.pointsScore * 0.20) +
                    (c.bottomScore * 0.15) +
                    (c.globalScore * 0.05)
            }))
            .sort((a, b) => b.finalScore - a.finalScore);

        best = fallback[0];
    }

    if (!best || !best.card) {
        return {
            card: cardsDb[0],
            score: 0
        };
    }

    return {
        card: best.card,
        score:
            best.finalScore ??
            (
                (best.colorScore * 0.15) +
                (best.symbolScore * 0.45) +
                (best.pointsScore * 0.20) +
                (best.bottomScore * 0.15) +
                (best.globalScore * 0.05)
            ),
        debug: {
            colorScore: best.colorScore,
            symbolScore: best.symbolScore,
            pointsScore: best.pointsScore,
            bottomScore: best.bottomScore,
            globalScore: best.globalScore
        }
    };
}
function enrichCandidate(querySig, card) {
    const cardSig = card.signature || card;

    const qColor = getScanPart(querySig, "color")?.color;
    const cColor = getScanPart(cardSig, "color")?.color;

    let colorScore = 0;

    if (qColor && cColor) {
        const d =
            Math.abs(qColor[0] - cColor[0]) +
            Math.abs(qColor[1] - cColor[1]) +
            Math.abs(qColor[2] - cColor[2]);

        colorScore = 1 / (1 + d / 50);
    }

    const symbolScore = meanStdScore(
        getScanPart(querySig, "symbol"),
        getScanPart(cardSig, "symbol")
    );

    const bottomScore = patchScore(
        getScanPart(querySig, "bottom"),
        getScanPart(cardSig, "bottom")
    );

    const globalScore = patchScore(
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
            (c.symbolScore * 0.55) +
            (c.bottomScore * 0.20) +
            (c.globalScore * 0.10)
    }));

    scored.sort((a, b) => b.finalScore - a.finalScore);

    return scored[0];
}

function matchSignature(querySig, cardsDb) {
    if (!querySig || !cardsDb || !cardsDb.length) {
        return {
            card: cardsDb ? cardsDb[0] : null,
            score: 0
        };
    }

    let candidates = cardsDb.map(card =>
        enrichCandidate(querySig, card)
    );

    // -----------------
    // COLOR
    // -----------------
    let stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });

    // -----------------
    // SYMBOL
    // -----------------
    const detectedSymbol =
        getScanPart(querySig, "symbol")?.name || null;

    const detectedSymbolScore =
        Number(getScanPart(querySig, "symbol")?.score || 0);

    let stepSymbol = stepColor;

    if (detectedSymbol && detectedSymbolScore >= 0.50) {
        const filtered = stepColor.filter(c => {
            if (!c.card.symbol) return false;

            return (
                c.card.symbol.toUpperCase().trim() ===
                detectedSymbol.toUpperCase().trim()
            );
        });

        if (filtered.length > 0) {
            stepSymbol = filtered;
            console.log("SYMBOL EXACT =", detectedSymbol);
        } else {
            console.log("SYMBOL FILTER FAILED → keep color");
        }
    } else {
        console.log("SYMBOL UNKNOWN");
    }

    // -----------------
    // BOTTOM
    // -----------------
    let stepBottom = keepBestBy(stepSymbol, "bottomScore", {
        keepTop: 4,
        ratio: 0.97,
        minKeep: 1
    });

    // -----------------
    // GLOBAL
    // -----------------
    let stepGlobal = keepBestBy(stepBottom, "globalScore", {
        keepTop: 2,
        ratio: 0.97,
        minKeep: 1
    });

    let best = finalTieBreak(stepGlobal);

    // -----------------
    // fallback
    // -----------------
    if (!best) {
        const fallback = [...candidates]
            .map(c => ({
                ...c,
                     finalScore:
                (c.colorScore * 0.15) +
                (c.symbolScore * 0.55) +
                (c.bottomScore * 0.20) +
                (c.globalScore * 0.10)
            }))
            .sort((a, b) => b.finalScore - a.finalScore);

        best = fallback[0];
    }

    if (!best || !best.card) {
        return {
            card: cardsDb[0],
            score: 0
        };
    }

    return {
        card: best.card,
        score:
            best.finalScore ??
            (
                best.colorScore * 0.10 +
                best.symbolScore * 0.30 +
                best.bottomScore * 0.45 +
                best.globalScore * 0.15
            ),
        debug: {
            colorScore: best.colorScore,
            symbolScore: best.symbolScore,
            bottomScore: best.bottomScore,
            globalScore: best.globalScore
        }
    };
}
