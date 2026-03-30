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

    const vecScore = vectorA && vectorB ? similarityScore(vectorA, vectorB) : 0;
    const statsScore = meanStdScore(a, b);

    if (vectorA && vectorB) {
        return (vecScore * 0.75) + (statsScore * 0.25);
    }

    return statsScore;
}

function getScanPart(signature, part) {
    if (!signature) return null;
    if (signature.scan && signature.scan[part] != null) return signature.scan[part];
    if (signature[part] != null) return signature[part];
    return null;
}

function getReliableDetectedPoints(querySig) {
    const points = getScanPart(querySig, "points");
    if (!points || points.digit == null) return null;

    const digit = Number(points.digit);
    if (!isFinite(digit)) return null;

    const score = Number(points.score ?? points.points_score ?? 0);
    const gap = Number(points.gap ?? points.points_gap ?? 0);

    if (score < 0.70) return null;
    if (gap < 0.02) return null;

    return digit;
}

function getDetectedSymbolInfo(querySig) {
    const symbol = getScanPart(querySig, "symbol") || {};
    const rawName = symbol.raw_name || symbol.name || null;
    const score = Number(symbol.score ?? 0);
    const gap = Number(symbol.gap ?? 0);
    const topCandidates = Array.isArray(symbol.top_candidates) ? symbol.top_candidates : [];
    const reliable = Boolean(rawName && score >= 0.62 && gap >= 0.05);

    return {
        rawName,
        name: reliable ? rawName : null,
        score,
        gap,
        reliable,
        topCandidates
    };
}

function pointsMatchScore(querySig, card) {
    const detectedPoints = getReliableDetectedPoints(querySig);

    if (detectedPoints == null) return 0.50;

    const cardPoints = Number(card.points);
    if (!isFinite(cardPoints)) return 0.50;

    if (cardPoints === detectedPoints) return 1.00;
    if (Math.abs(cardPoints - detectedPoints) === 1) return 0.15;
    return 0.00;
}

function symbolMatchScore(symbolInfo, card) {
    if (!symbolInfo || !symbolInfo.reliable || !symbolInfo.name) return 0.50;

    const cardSymbol = String(card.symbol || "").toUpperCase().trim();
    const detectedSymbol = String(symbolInfo.name || "").toUpperCase().trim();

    if (!cardSymbol || !detectedSymbol) return 0.00;
    return cardSymbol === detectedSymbol ? 1.00 : 0.00;
}

function enrichCandidate(querySig, card, symbolInfo) {
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

    const symbolScore = symbolMatchScore(symbolInfo, card);
    const pointsScore = pointsMatchScore(querySig, card);
    const bottomScore = patchScore(getScanPart(querySig, "bottom"), getScanPart(cardSig, "bottom"));
    const globalScore = patchScore(getScanPart(querySig, "global"), getScanPart(cardSig, "global"));

    const cardSymbol = String(card.symbol || "").toUpperCase().trim();
    const symbolExactMatch = Boolean(
        symbolInfo && symbolInfo.name && cardSymbol === String(symbolInfo.name).toUpperCase().trim()
    );

    return {
        card,
        colorScore,
        symbolScore,
        pointsScore,
        bottomScore,
        globalScore,
        symbolExactMatch
    };
}

function keepBestBy(candidates, key, options = {}) {
    const { keepTop = 8, ratio = 0.92, minKeep = 2 } = options;
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

function computeFinalScore(candidate, options = {}) {
    if (options.useReliableSymbol) {
        return (
            (candidate.colorScore * 0.18) +
            (candidate.symbolScore * 0.33) +
            (candidate.pointsScore * 0.04) +
            (candidate.bottomScore * 0.30) +
            (candidate.globalScore * 0.15)
        );
    }

    return (
        (candidate.colorScore * 0.26) +
        (candidate.pointsScore * 0.05) +
        (candidate.bottomScore * 0.44) +
        (candidate.globalScore * 0.25)
    );
}

function scoreFinalCandidates(candidates, options = {}) {
    return [...candidates]
        .map(c => ({ ...c, finalScore: computeFinalScore(c, options) }))
        .sort((a, b) => b.finalScore - a.finalScore);
}

function summarizeCandidates(candidates, limit = 5) {
    return [...candidates].slice(0, limit).map(c => ({
        id: c.card?.id,
        couleur: c.card?.couleur,
        symbol: c.card?.symbol,
        points: c.card?.points,
        colorScore: Number(c.colorScore ?? 0),
        symbolScore: Number(c.symbolScore ?? 0),
        pointsScore: Number(c.pointsScore ?? 0),
        bottomScore: Number(c.bottomScore ?? 0),
        globalScore: Number(c.globalScore ?? 0),
        boostedBottom: Number(c.boostedBottom ?? 0),
        finalScore: Number(c.finalScore ?? 0),
        symbolExactMatch: Boolean(c.symbolExactMatch)
    }));
}

function matchSignature(querySig, cardsDb) {
    if (!querySig || !cardsDb || !cardsDb.length) {
        return { card: cardsDb ? cardsDb[0] : null, score: 0 };
    }

    const symbolInfo = getDetectedSymbolInfo(querySig);
    const candidates = cardsDb.map(card => enrichCandidate(querySig, card, symbolInfo));

    const stepColor = keepBestBy(candidates, "colorScore", {
        keepTop: 12,
        ratio: 0.90,
        minKeep: 4
    });

    let stepSymbol = stepColor;
    let symbolFilterApplied = false;

    if (symbolInfo.reliable && symbolInfo.name) {
        const filtered = stepColor.filter(c => c.symbolExactMatch);
        if (filtered.length > 0) {
            stepSymbol = filtered;
            symbolFilterApplied = true;
            console.log("SYMBOL EXACT =", symbolInfo.name, "matches =", filtered.length);
        } else {
            console.log("SYMBOL RELIABLE BUT NO MATCHING CARD -> keep color");
        }
    } else {
        console.log("SYMBOL NOT RELIABLE", symbolInfo);
    }

    const stepPoints = stepSymbol.map(c => ({
        ...c,
        boostedBottom: (c.bottomScore * 0.90) + (c.pointsScore * 0.10)
    }));

    const stepBottom = keepBestBy(stepPoints, "boostedBottom", {
        keepTop: 4,
        ratio: 0.97,
        minKeep: 1
    });

    const stepGlobal = keepBestBy(stepBottom, "globalScore", {
        keepTop: 2,
        ratio: 0.97,
        minKeep: 1
    });

    const finalOptions = { useReliableSymbol: symbolFilterApplied };
    const scoredFinal = scoreFinalCandidates(stepGlobal, finalOptions);
    let best = scoredFinal[0] || null;

    if (!best) {
        best = scoreFinalCandidates(candidates, finalOptions)[0] || null;
    }

    if (!best || !best.card) {
        return { card: cardsDb[0], score: 0 };
    }

    const debug = {
        detectedSymbol: symbolInfo,
        symbolFilterApplied,
        steps: {
            afterColor: summarizeCandidates(stepColor, 5),
            afterSymbol: summarizeCandidates(stepSymbol, 5),
            afterBottom: summarizeCandidates(stepBottom, 5),
            afterGlobal: summarizeCandidates(stepGlobal, 5),
            final: summarizeCandidates(scoredFinal, 5)
        },
        best: {
            id: best.card?.id,
            colorScore: best.colorScore,
            symbolScore: best.symbolScore,
            pointsScore: best.pointsScore,
            bottomScore: best.bottomScore,
            globalScore: best.globalScore,
            finalScore: best.finalScore
        }
    };

    console.log("MATCH DEBUG =", debug);

    return {
        card: best.card,
        score: best.finalScore ?? computeFinalScore(best, finalOptions),
        debug
    };
}
