// Génère MS-VICRegL_synthese.docx — synthèse problématiques + résultats
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');

const BLUE = '2952cc', RED = 'c0392b', GREEN = '1e8449', ORANGE = 'b9770e';
const GREY = '5d6d7e', DARK = '1c2833';

// ---- helpers ----
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 120 }, children: [new TextRun({ text: t, bold: true, color: DARK })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 }, children: [new TextRun({ text: t, bold: true, color: BLUE })] });
const P = (runs, opts = {}) => new Paragraph({ spacing: { after: 100, line: 276 }, alignment: AlignmentType.JUSTIFIED, children: Array.isArray(runs) ? runs : [new TextRun(runs)], ...opts });
const T = (text, o = {}) => new TextRun({ text, ...o });
const bullet = (runs) => new Paragraph({ numbering: { reference: 'bul', level: 0 }, spacing: { after: 60, line: 268 }, children: Array.isArray(runs) ? runs : [new TextRun(runs)] });

function shadeCell(text, { bg, bold, color, align, w }) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: bg ? { type: ShadingType.CLEAR, color: 'auto', fill: bg } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!bold, color: color || DARK, size: 19 })]
    })]
  });
}
function tableRow(cells, colW, { bg, bold, colors } = {}) {
  return new TableRow({
    children: cells.map((c, i) => shadeCell(c, {
      bg, bold, w: colW[i],
      color: colors ? colors[i] : undefined,
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER
    }))
  });
}
function makeTable(header, rows, colW, rowOpts) {
  const total = colW.reduce((a, b) => a + b, 0);
  return new Table({
    columnWidths: colW,
    width: { size: total, type: WidthType.DXA },
    rows: [
      tableRow(header, colW, { bg: DARK, bold: true, colors: header.map(() => 'FFFFFF') }),
      ...rows.map((r, idx) => tableRow(r.cells, colW, {
        bg: idx % 2 ? 'F4F6F7' : 'FFFFFF', bold: r.bold, colors: r.colors
      }))
    ]
  });
}

const COL = { size: 9360, type: WidthType.DXA };

const doc = new Document({
  creator: 'MS-VICRegL',
  title: 'MS-VICRegL — Synthèse',
  numbering: {
    config: [{
      reference: 'bul', levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 200 } } }
      }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Calibri', size: 21, color: DARK } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, color: DARK }, paragraph: { spacing: { before: 260, after: 120 } } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 23, bold: true, color: BLUE }, paragraph: { spacing: { before: 180, after: 80 } } }
    ]
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 } } },
    children: [
      // ===== TITRE =====
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: 'Identification bactérienne par MALDI-TOF', bold: true, size: 34, color: DARK })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Un CNN 1D auto-supervisé (VICRegL) invariant aux artefacts de centre', size: 26, color: BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: 'Synthèse des problématiques et des résultats', italics: true, size: 22, color: GREY })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, border: { bottom: { color: BLUE, size: 12, style: BorderStyle.SINGLE, space: 8 } }, children: [new TextRun({ text: 'Jeu DRIAMS (4 centres hospitaliers) · comparaison au pipeline classique MSclassifR', size: 19, color: GREY })] }),

      // ===== 1. CONTEXTE =====
      H1('1. Contexte et problématique'),
      P([
        T('L’identification bactérienne par spectrométrie MALDI-TOF est routinière, mais les modèles souffrent d’un '),
        T('effet de centre', { bold: true }),
        T(' : calibration m/z, ligne de base, gain du détecteur et matrice varient d’un hôpital et d’un appareil à l’autre. Un classifieur entraîné dans un centre '),
        T('se dégrade sur un autre', { italics: true }),
        T(', ce qui limite le déploiement multi-sites.')
      ]),
      P([
        T('Le pipeline de référence (type '),
        T('MSclassifR', { bold: true }),
        T(') neutralise ces artefacts par un pré-traitement lourd fait main (stabilisation de variance, ondelettes, SNIP, normalisation TIC, binning 3 Da, correction de lot). '),
        T('Question de ce travail :', { bold: true, color: BLUE }),
        T(' peut-on plutôt '),
        T('apprendre l’invariance aux artefacts', { bold: true }),
        T(', à partir de spectres quasi-bruts, sans ce pré-traitement expert ?')
      ]),

      // ===== 2. APPROCHE =====
      H1('2. Approche proposée'),
      P([
        T('Un encodeur '),
        T('CNN 1D (ResNet-1D)', { bold: true }),
        T(' est pré-entraîné en '),
        T('auto-supervision VICRegL', { bold: true }),
        T(' — sans étiquettes. Les augmentations '),
        T('simulent les artefacts MALDI', { italics: true }),
        T(' (déformation de calibration m/z, ligne de base, gain, bruit, perte de pics) : le réseau apprend à ignorer ces nuisances tout en préservant la structure discriminante des pics (biomarqueurs) via le critère local de VICRegL.')
      ]),
      P([T('Entrée :', { bold: true }), T(' spectre brut ré-échantillonné (2000–20000 Da, 6000 points) + normalisation TIC seulement. ', {}),
        T('Évaluation :', { bold: true }), T(' sonde linéaire sur features gelées, comparée à un Random Forest sur '), T('binned_6000', { font: 'Consolas', size: 19 }), T(' (représentation DRIAMS standard, pré-traitement complet).')]),
      P([T('Données DRIAMS :', { bold: true }), T(' 4 centres — B (Bâle-Land), C (Aarau), D (Viollier), A (Bâle). Étiquettes '), T('Biotyper (MALDI)', { bold: true, color: ORANGE }), T('. Métrique principale : '), T('balanced accuracy', { italics: true }), T(' (classes déséquilibrées).')]),

      // ===== 3. RÉSULTATS =====
      H1('3. Résultats — vue d’ensemble'),
      makeTable(
        ['Expérience', 'Résultat clé', 'Verdict'],
        [
          { cells: ['ID espèces cross-centre B↔C', 'VICRegL 0,95–0,99 (stable) vs RF 0,68–1,00 (erratique) — ~8× plus stable', 'Favorable*'], colors: [DARK, DARK, GREEN], bold: false },
          { cells: ['Espèces proches (Enterobacter, Strepto, Klebsiella)', 'VICRegL > RF sur les 3 groupes (McNemar p<0,01)', 'Favorable*'], colors: [DARK, DARK, GREEN] },
          { cells: ['Résistance antibiotique (AMR)', 'RF bat VICRegL, aucun gain de stabilité', 'Défavorable'], colors: [DARK, DARK, RED] },
          { cells: ['Hold-out centre D (jamais vu)', 'RF 0,935 > VICRegL 0,886 — renversement', 'Défavorable'], colors: [DARK, DARK, RED] },
          { cells: ['D ajouté au pré-entraînement (non-labellisé)', '0,886 → 0,915 : écart au RF divisé par 2', 'Partiel'], colors: [DARK, DARK, ORANGE] }
        ],
        [3200, 4360, 1800]
      ),
      new Paragraph({ spacing: { before: 80, after: 120 }, children: [new TextRun({ text: '* dans le cadre où les deux centres ont été vus (non-labellisés) au pré-entraînement — voir §4.', italics: true, size: 18, color: GREY })] }),

      // ===== 4. LE POINT CRITIQUE =====
      H1('4. Le point critique : adaptation ≠ généralisation'),
      P([
        T('Le résultat marquant « ~8× plus stable » (B↔C) est obtenu quand '),
        T('les deux centres ont été vus, non-labellisés, au pré-entraînement', { bold: true }),
        T(' : c’est une '),
        T('adaptation de domaine à cible disponible', { italics: true }),
        T(', pas de la généralisation à un centre neuf. Le '),
        T('test strict', { bold: true }),
        T(' — centre D tenu totalement à l’écart du SSL — '),
        T('inverse la conclusion', { bold: true, color: RED }),
        T(' : le pipeline classique l’emporte.')
      ]),
      makeTable(
        ['Condition (éval. sur D)', 'Encodeur B∪C\n(D jamais vu)', 'Encodeur B∪C∪D\n(D vu non-labellisé)', 'RF classique'],
        [
          { cells: ['(B+C) → D  (centre inédit)', '0,886', '0,915', '0,935'], colors: [DARK, RED, ORANGE, GREEN], bold: true },
          { cells: ['Plafond D → D (sonde sur D)', '0,914', '0,931', '—'] },
          { cells: ['B+C en intra-domaine (plafond)', '0,988', '0,990', '0,889'] },
          { cells: ['AUROC « quel centre ? » sur features', '0,997', '0,999', '—'], colors: [DARK, RED, RED, DARK] }
        ],
        [3360, 2000, 2000, 2000]
      ),
      new Paragraph({ spacing: { before: 100, after: 80 }, children: [new TextRun({ text: 'Balanced accuracy ; 10 espèces communes B∩C∩D ; test = 8797 spectres de D.', italics: true, size: 18, color: GREY })] }),
      P([
        T('Diagnostic :', { bold: true, color: BLUE }),
        T(' l’AUROC « quel centre ? » reste ~1,0 même après avoir vu D — les features '),
        T('encodent encore parfaitement le centre', { bold: true }),
        T('. L’invariance revendiquée '),
        T('n’est pas atteinte', { bold: true, color: RED }),
        T(' ; ce qui fonctionnait en B↔C relevait de l’adaptation. Voir D en SSL enrichit la représentation (gain réel, sans labels) mais '),
        T('ne restaure ni la parité avec le RF ni l’invariance', { italics: true }),
        T('.')
      ]),

      // ===== 5. FORCES / LIMITES =====
      H1('5. Forces et limites'),
      H2('Forces'),
      bullet([T('Pas de pré-traitement expert', { bold: true }), T(' : spectre quasi-brut + TIC ; plafond intra-domaine élevé (0,99).')]),
      bullet([T('Extension d’un centre sans aucun label', { bold: true }), T(' : l’ajouter au SSL (non-supervisé) réduit de moitié l’écart au pipeline classique.')]),
      bullet([T('Diagnostic rigoureux', { bold: true }), T(' : décomposition plafond/transfert, classifieur de domaine, McNemar apparié, IC bootstrap.')]),
      H2('Limites'),
      bullet([T('Circularité des étiquettes', { bold: true, color: ORANGE }), T(' : DRIAMS est labellisé Biotyper (MALDI), pas NGS — 0 Shigella dans B+C. On ne peut pas prétendre « battre le MALDI » avec des labels MALDI.')]),
      bullet([T('Invariance non atteinte', { bold: true }), T(' : AUROC domaine ~1,0 ; échec sur un centre vraiment inédit.')]),
      bullet([T('AMR', { bold: true }), T(' : les augmentations (perte de pics) effacent le signal faible que la résistance requiert.')]),

      // ===== 6. CONCLUSION =====
      H1('6. Conclusion et perspectives'),
      P([
        T('Le pipeline est un '),
        T('prototype de méthode prometteur', { bold: true }),
        T(', pas un système supérieur au classique : il excelle en intra-domaine et en adaptation, mais '),
        T('l’invariance apprise ne generalise pas encore', { italics: true }),
        T(' à un centre inédit, où le pré-traitement fait main reste devant.')
      ]),
      P([T('Deux leviers pour la suite :', { bold: true, color: BLUE })]),
      bullet([T('Terme d’invariance de centre explicite', { bold: true }), T(' dans la loss (domaine-adversarial / CORAL) — attaque directement l’AUROC ~1,0, cause racine de l’échec.')]),
      bullet([T('Données à étiquettes moléculaires (WGS)', { bold: true }), T(' — seul moyen de lever la circularité et de tester la vraie hypothèse : l’invariance capte-t-elle un signal que les labels MALDI ratent ?')]),
      P([
        T('Ce dernier point ouvre une '),
        T('collaboration naturelle', { bold: true, color: GREEN }),
        T(' avec les travaux sur données MALDI appariées à des labels moléculaires (identification fine de sous-espèces, rejet par confiance).')
      ]),
      new Paragraph({ spacing: { before: 200 }, border: { top: { color: 'D5D8DC', size: 8, style: BorderStyle.SINGLE, space: 6 } }, children: [new TextRun({ text: 'Reproductibilité : encodeur ResNet-1D, PyTorch MPS (Apple M3 Pro). Résultats régénérables via scripts/08_holdout_D.py et 09_ssl_BCD.py.', italics: true, size: 17, color: GREY })] })
    ]
  }]
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync('MS-VICRegL_synthese.docx', buf);
  console.log('OK -> MS-VICRegL_synthese.docx');
});
