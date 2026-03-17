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

    if (signature.scan && signature.scan[part] != null)
        return signature.scan[part];

    if (signature[part] != null)
        return signature[part];

    return null;
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
    (c.colorScore * 0.05) +
    (c.symbolScore * 0.10) +
    (c.bottomScore * 0.70) +
    (c.globalScore * 0.15)
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
    // BOTTOM FIRST
    // -----------------
    // La zone violette du bas devient le critère principal
    // pour départager les cartes d'une même couleur.
    let stepBottom = keepBestBy(stepColor, "bottomScore", {
        keepTop: 5,
        ratio: 0.985,
        minKeep: 2
    });

    // -----------------
    // SYMBOL NAME FROM SCAN
    // -----------------
    // Le symbole du haut devient une aide.
    // Il ne doit plus casser la détection si ce n'est pas fiable.
    const detectedSymbol =
        getScanPart(querySig, "symbol")?.name || null;

    const detectedSymbolScore =
        Number(getScanPart(querySig, "symbol")?.score || 0);

    let stepSymbol = stepBottom;

    // Filtre fort uniquement si le symbole paraît assez fiable
    if (detectedSymbol && detectedSymbolScore >= 0.45) {
        const filtered = stepBottom.filter(c => {
            if (!c.card.symbol) return true;

            return (
                c.card.symbol &&
                detectedSymbol &&
                c.card.symbol.toUpperCase().trim() ===
                detectedSymbol.toUpperCase().trim()
            );
        });

        // On n'applique le filtre que s'il reste au moins 1 candidat
        if (filtered.length >= 1) {
            stepSymbol = filtered;
            console.log("SYMBOL HELP =", detectedSymbol);
        } else {
            console.log("SYMBOL FILTER FAILED → keep bottom result");
        }
    }

    // Si le symbole n'est pas assez fiable, on ne coupe pas brutalement.
    // On garde juste les meilleurs selon le score symbole, mais après le bottom.
    if (stepSymbol === stepBottom) {
        stepSymbol = keepBestBy(stepBottom, "symbolScore", {
            keepTop: 3,
            ratio: 0.97,
            minKeep: 1
        });
    }

    // -----------------
    // GLOBAL
    // -----------------
    let stepGlobal = keepBestBy(stepSymbol, "globalScore", {
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
                    (c.colorScore * 0.05) +
                    (c.symbolScore * 0.10) +
                    (c.bottomScore * 0.70) +
                    (c.globalScore * 0.15)
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
                best.colorScore * 0.05 +
                best.symbolScore * 0.10 +
                best.bottomScore * 0.70 +
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
