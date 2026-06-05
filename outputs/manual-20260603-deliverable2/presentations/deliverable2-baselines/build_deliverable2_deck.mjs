import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/keshavperiwal/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

const ROOT = "/Users/keshavperiwal/Documents Local/Semester 2/Project Time Series/Data";
const WORKSPACE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(ROOT, "results", "Deliverable2_Baseline_Models_Presentation.pptx");
const PREVIEW_DIR = path.join(WORKSPACE, "preview");
const LAYOUT_DIR = path.join(WORKSPACE, "layout");
const FIGURES = path.join(ROOT, "results", "figures");
const TABLES = path.join(ROOT, "results", "tables");
const ASSETS = path.join(ROOT, "results", "template_assets");
const W = 1280;
const H = 720;

const C = {
  navy: "#003A63",
  navy2: "#0B4F71",
  teal: "#2F586E",
  pale: "#EEF3F7",
  line: "#CBD5DE",
  ink: "#1E2A32",
  muted: "#5F6B76",
  red: "#B45555",
  blue: "#3F759B",
  orange: "#D66922",
  green: "#3A8F58",
  white: "#FFFFFF",
};

function readCsv(file) {
  const text = fsSync.readFileSync(file, "utf8").trim();
  const [header, ...lines] = text.split(/\r?\n/);
  const keys = header.split(",");
  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(keys.map((key, index) => [key, values[index]]));
  });
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function auc(value) {
  return Number(value).toFixed(3);
}

function modelLabel(model) {
  return {
    random_forest: "Random Forest",
    linear_svm: "SVM linear",
    svm_rbf: "SVM RBF",
    resnet1d: "1D-ResNet",
    lstm: "LSTM",
    gru: "GRU",
    patchtst: "PatchTST",
  }[model] || model;
}

function familyLabel(family) {
  return family === "feature_classical" ? "Feature vector" : "Feature vector NN";
}

function rowsForScope(summary, scope) {
  const order = ["random_forest", "linear_svm", "svm_rbf", "resnet1d", "lstm", "gru", "patchtst"];
  return summary
    .filter((row) => row.scope === scope)
    .sort((a, b) => order.indexOf(a.model) - order.indexOf(b.model));
}

function scopeTitle(scope) {
  return {
    left: "Left Foot Classification Report",
    right: "Right Foot Classification Report",
    combined: "Combined Foot Classification Report",
    enhanced: "Enhanced Feature Set Classification Report",
  }[scope] || `${scope[0].toUpperCase()}${scope.slice(1)} Classification Report`;
}

function addHeader(ctx, slide, title, section = "Deliverable 2 - Baseline Models") {
  ctx.addShape(slide, { x: 0, y: 0, width: W, height: 104, fill: C.navy, line: { fill: C.navy, width: 0 } });
  ctx.addText(slide, {
    text: title,
    x: 28,
    y: 36,
    width: 850,
    height: 42,
    fontSize: 28,
    bold: true,
    color: C.white,
    typeface: "Arial",
  });
  ctx.addText(slide, {
    text: section,
    x: 28,
    y: 110,
    width: 260,
    height: 30,
    fontSize: 12,
    bold: true,
    color: C.white,
    fill: C.navy,
    insets: { left: 12, right: 8, top: 7, bottom: 0 },
  });
  ctx.addShape(slide, { x: 0, y: 126, width: W, height: 3, fill: C.teal, line: { fill: C.teal, width: 0 } });
}

function addFooter(ctx, slide, num, total) {
  ctx.addShape(slide, { x: 0, y: 686, width: W, height: 34, fill: "#DDE5EC", line: { fill: C.line, width: 0.5 } });
  ctx.addText(slide, {
    text: "Deliverable 2 - PD Gait Baseline Models  |  FAU Erlangen-Nürnberg",
    x: 14,
    y: 694,
    width: 900,
    height: 18,
    fontSize: 9,
    color: C.muted,
  });
  ctx.addText(slide, {
    text: `${num}/${total}`,
    x: 1210,
    y: 694,
    width: 55,
    height: 18,
    fontSize: 9,
    color: C.muted,
    align: "right",
  });
}

async function addLogos(ctx, slide) {
  ctx.addText(slide, {
    text: "Friedrich-Alexander-Universität\nTechnische Fakultät",
    x: 54,
    y: 31,
    width: 270,
    height: 46,
    fontSize: 13,
    bold: true,
    color: C.navy,
  });
  await ctx.addImage(slide, {
    path: path.join(ASSETS, "image2.png"),
    x: 905,
    y: 32,
    width: 190,
    height: 66,
    fit: "contain",
  });
  ctx.addText(slide, {
    text: "FAU",
    x: 1100,
    y: 27,
    width: 130,
    height: 62,
    fontSize: 42,
    bold: true,
    color: C.navy,
    align: "center",
  });
  ctx.addShape(slide, { x: 54, y: 126, width: 1170, height: 5, fill: C.teal, line: { fill: C.teal, width: 0 } });
}

function addKpi(ctx, slide, label, value, x, y, color = C.navy) {
  ctx.addShape(slide, { x, y, width: 176, height: 94, fill: C.white, line: { fill: C.line, width: 1 } });
  ctx.addText(slide, { text: value, x: x + 14, y: y + 15, width: 150, height: 34, fontSize: 27, bold: true, color });
  ctx.addText(slide, { text: label, x: x + 14, y: y + 55, width: 150, height: 25, fontSize: 11, color: C.muted });
}

function addBullets(ctx, slide, lines, x, y, width, height, fontSize = 18) {
  ctx.addText(slide, {
    text: lines.map((line) => `• ${line}`).join("\n"),
    x,
    y,
    width,
    height,
    fontSize,
    color: C.ink,
    typeface: "Arial",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

function addTable(ctx, slide, columns, rows, x, y, widths, rowH = 39, fontSize = 14) {
  let cx = x;
  columns.forEach((col, index) => {
    ctx.addShape(slide, { x: cx, y, width: widths[index], height: rowH, fill: C.navy, line: { fill: C.white, width: 1 } });
    ctx.addText(slide, {
      text: col,
      x: cx + 6,
      y: y + 10,
      width: widths[index] - 12,
      height: rowH - 8,
      fontSize,
      bold: true,
      color: C.white,
      align: index === 0 ? "left" : "center",
    });
    cx += widths[index];
  });
  rows.forEach((row, r) => {
    cx = x;
    const fill = r % 2 === 0 ? "#F4F7FA" : C.white;
    row.forEach((cell, index) => {
      ctx.addShape(slide, { x: cx, y: y + rowH * (r + 1), width: widths[index], height: rowH, fill, line: { fill: C.line, width: 1 } });
      ctx.addText(slide, {
        text: String(cell),
        x: cx + 6,
        y: y + rowH * (r + 1) + 10,
        width: widths[index] - 12,
        height: rowH - 8,
        fontSize,
        bold: index === 0,
        color: C.ink,
        align: index === 0 ? "left" : "center",
      });
      cx += widths[index];
    });
  });
}

function classificationRows(scopeRows) {
  return scopeRows.map((row) => [
    modelLabel(row.model),
    familyLabel(row.family),
    percent(row.test_accuracy_mean),
    percent(row.test_sensitivity_mean),
    percent(row.test_specificity_mean),
    auc(row.test_auc_mean),
  ]);
}

async function addImage(ctx, slide, file, x, y, width, height) {
  await ctx.addImage(slide, { path: path.join(FIGURES, file), x, y, width, height, fit: "contain" });
}

function addProtocolRail(ctx, slide, items, x, y) {
  items.forEach((item, i) => {
    const top = y + i * 88;
    ctx.addShape(slide, { x, y: top, width: 52, height: 52, fill: C.navy, line: { fill: C.navy, width: 0 } });
    ctx.addText(slide, { text: String(i + 1).padStart(2, "0"), x: x + 5, y: top + 14, width: 42, height: 22, fontSize: 17, bold: true, color: C.white, align: "center" });
    ctx.addText(slide, { text: item.title, x: x + 68, y: top, width: 430, height: 24, fontSize: 18, bold: true, color: C.navy });
    ctx.addText(slide, { text: item.body, x: x + 68, y: top + 28, width: 470, height: 44, fontSize: 12, color: C.muted });
  });
}

async function build() {
  const artifact = await importArtifactTool(WORKSPACE);
  const { Presentation, PresentationFile } = artifact;
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const ctx = createSlideContext(artifact, { slideSize: { width: W, height: H }, workspaceDir: WORKSPACE });
  const summary = readCsv(path.join(TABLES, "deliverable2_model_summary.csv"));
  const best = readCsv(path.join(TABLES, "deliverable2_best_models_by_scope.csv"));
  const cm = readCsv(path.join(TABLES, "deliverable2_best_confusion_matrices.csv"));
  const reportScopes = ["left", "right", "combined", "enhanced"];
  const totalSlides = 12;
  let n = 1;

  let slide = presentation.slides.add();
  await ctx.addImage(slide, { path: path.join(ASSETS, "image3.jpg"), x: 0, y: 0, width: W, height: H, fit: "cover" });
  ctx.addShape(slide, { x: 0, y: 0, width: W, height: H, fill: "#003A6399", line: { fill: "#003A6300", width: 0 } });
  ctx.addText(slide, { text: "Deliverable 2", x: 70, y: 205, width: 500, height: 34, fontSize: 18, color: C.white });
  ctx.addText(slide, { text: "Baseline Models for Parkinson's Disease Gait Detection", x: 70, y: 250, width: 850, height: 95, fontSize: 42, bold: true, color: C.white });
  ctx.addText(slide, { text: "Random Forest, SVM, 1D-ResNet, LSTM, GRU, and PatchTST on fold-local feature vectors", x: 72, y: 365, width: 850, height: 34, fontSize: 18, color: "#E8F0F5" });
  const bestAuc = Math.max(...best.map((row) => Number(row.test_auc_mean)));
  addKpi(ctx, slide, "best held-out AUC", auc(bestAuc), 72, 470, C.navy);
  addKpi(ctx, slide, "test subjects per full CV", "100", 270, 470, C.navy);
  addKpi(ctx, slide, "views reported", "4", 468, 470, C.navy);
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  await addLogos(ctx, slide);
  ctx.addText(slide, { text: "Deliverable 2 Requirements", x: 57, y: 255, width: 760, height: 52, fontSize: 30, bold: true, color: C.navy });
  addBullets(ctx, slide, [
    "Train Random Forest, SVM linear, and SVM RBF with validation-selected hyperparameters.",
    "Extract training, validation, and test feature matrices separately inside each fold.",
    "Normalize the feature matrix using z-score statistics learned from the training split.",
    "Report left-foot, right-foot, and combined-foot classification results.",
    "Add an enhanced feature view for the stronger TSFresh + gait/asymmetry/wavelet baseline comparison.",
    "Include Accuracy, Sensitivity, Specificity, and AUC.",
    "Add confusion matrices for the best-performing model in each view.",
    "Include 1D-ResNet, LSTM, GRU, and PatchTST as feature-vector neural baselines.",
  ], 58, 320, 1050, 245, 18);
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Evaluation Protocol");
  addProtocolRail(ctx, slide, [
    { title: "Use provided stratified folds", body: "Each fold has 60 training, 20 validation, and 20 test subjects with balanced PD/control labels." },
    { title: "Extract split-local features", body: "Every fold writes separate training, validation, and test feature matrices from its own signal files." },
    { title: "Fit preprocessing inside each fold", body: "Median imputation, variance filtering, z-score scaling, and feature selection are fit on training only." },
    { title: "Select hyperparameters on validation", body: "RF and SVM grids are ranked by validation F1, then accuracy and AUC as tie-breakers to match the meeting-slide comparison." },
    { title: "Report held-out test performance", body: "Final fold metrics are computed on the untouched test split and averaged over the five folds." },
  ], 72, 155);
  addTable(ctx, slide, ["Split", "Control", "Patient", "Total"], [
    ["Training", "30", "30", "60"],
    ["Validation", "10", "10", "20"],
    ["Test", "10", "10", "20"],
  ], 720, 220, [170, 130, 130, 130], 46, 16);
  ctx.addText(slide, { text: "Positive class = Patient. Sensitivity = TP / (TP + FN). Specificity = TN / (TN + FP).", x: 720, y: 405, width: 460, height: 60, fontSize: 16, color: C.muted });
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Model Families Implemented");
  addTable(ctx, slide, ["Family", "Models", "Input", "Selection"], [
    ["Feature classical", "Random Forest", "TSFresh + engineered feature matrix", "Grid search on validation"],
    ["Feature classical", "SVM linear", "z-scored feature matrix", "C and k-best grid"],
    ["Feature classical", "SVM RBF", "z-scored feature matrix", "C, gamma, k-best grid"],
    ["Feature classical", "Enhanced view", "TSFresh + asymmetry + gait + wavelet", "same validation protocol"],
    ["Feature neural", "1D-ResNet", "selected feature vector as 1D signal", "best validation epoch"],
    ["Feature neural", "LSTM / GRU", "selected feature vector sequence", "best validation epoch"],
    ["Feature neural", "PatchTST", "patched feature vector", "best validation epoch"],
  ], 55, 160, [170, 180, 500, 280], 48, 13);
  ctx.addText(slide, { text: "All models now use fold-local feature vectors. Neural models use train-fitted imputation, z-score scaling, and SelectKBest before training.", x: 70, y: 545, width: 1080, height: 60, fontSize: 18, color: C.navy, bold: true });
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Overall Results");
  await addImage(ctx, slide, "deliverable2_metric_summary.png", 45, 155, 1190, 470);
  ctx.addText(slide, { text: "Fold-local feature-vector baselines remain strongest; feature-vector PatchTST is the best neural variant in the enhanced view.", x: 68, y: 625, width: 1050, height: 32, fontSize: 18, bold: true, color: C.navy });
  addFooter(ctx, slide, n++, totalSlides);

  for (const scope of reportScopes) {
    slide = presentation.slides.add();
    addHeader(ctx, slide, scopeTitle(scope));
    const scopeRows = classificationRows(rowsForScope(summary, scope));
    addTable(ctx, slide, ["Model", "Input", "Accuracy", "Sensitivity", "Specificity", "AUC"], scopeRows, 48, 165, [190, 180, 145, 145, 145, 115], 54, 14);
    const bestRow = best.find((row) => row.scope === scope);
    ctx.addShape(slide, { x: 1000, y: 165, width: 210, height: 168, fill: C.pale, line: { fill: C.line, width: 1 } });
    ctx.addText(slide, { text: "Best model", x: 1020, y: 184, width: 170, height: 24, fontSize: 15, bold: true, color: C.navy });
    ctx.addText(slide, { text: modelLabel(bestRow.model), x: 1020, y: 220, width: 170, height: 26, fontSize: 18, bold: true, color: C.ink });
    ctx.addText(slide, { text: `AUC ${auc(bestRow.test_auc_mean)}\nAcc ${percent(bestRow.test_accuracy_mean)}\nSens ${percent(bestRow.test_sensitivity_mean)}\nSpec ${percent(bestRow.test_specificity_mean)}`, x: 1020, y: 265, width: 170, height: 70, fontSize: 15, color: C.ink });
    addFooter(ctx, slide, n++, totalSlides);
  }

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Best Confusion Matrices");
  await addImage(ctx, slide, "deliverable2_best_confusion_matrices.png", 65, 145, 1150, 430);
  addTable(ctx, slide, ["View", "Best model", "TN", "FP", "FN", "TP"], cm.map((row) => [row.scope, modelLabel(row.model), row.tn, row.fp, row.fn, row.tp]), 210, 565, [150, 230, 85, 85, 85, 85], 22, 10);
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Interpretation");
  addTable(ctx, slide, ["View", "Best model", "Accuracy", "Sensitivity", "Specificity", "AUC"], best.map((row) => [
    row.scope,
    modelLabel(row.model),
    percent(row.test_accuracy_mean),
    percent(row.test_sensitivity_mean),
    percent(row.test_specificity_mean),
    auc(row.test_auc_mean),
  ]), 60, 175, [155, 230, 150, 160, 160, 120], 52, 15);
  addBullets(ctx, slide, [
    "Classical feature models are strongest under the strict fold-local feature protocol.",
    "Enhanced Linear-SVM reaches 79.0% mean held-out accuracy with sensitivity 84.0%, specificity 74.0%, and AUC 0.870.",
    "Enhanced Random Forest has the best held-out AUC among the classical models.",
    "Feature-vector PatchTST is the strongest neural variant on the enhanced scope.",
    "LSTM/GRU remain unstable with only 60 training subjects per fold.",
  ], 70, 410, 1060, 160, 19);
  addFooter(ctx, slide, n++, totalSlides);

  slide = presentation.slides.add();
  addHeader(ctx, slide, "Recommended Next Improvements");
  addBullets(ctx, slide, [
    "Try a small hyperparameter grid for feature-vector neural models after the fold-safe baseline is locked.",
    "Run full EfficientFCParameters extraction for all 100 fold subjects and compare it with the current curated TSFresh set.",
    "Use TSFresh's relevance filter as a secondary feature-selection experiment; the quick probe improved RBF-SVM AUC but did not beat Linear-SVM accuracy.",
    "Add cohort-stratified reporting because Ga, Si, and Ju were collected under different walking conditions.",
    "Inspect fold-level errors to identify subjects repeatedly confused across models.",
    "Keep Mamba separate for a later deliverable, after these non-Mamba baselines are locked.",
  ], 70, 175, 940, 235, 21);
  ctx.addShape(slide, { x: 70, y: 460, width: 1050, height: 1, fill: C.line, line: { fill: C.line, width: 0 } });
  ctx.addText(slide, { text: "Deliverable 2 artifacts generated", x: 70, y: 495, width: 420, height: 28, fontSize: 20, bold: true, color: C.navy });
  addBullets(ctx, slide, [
    "results/tables/deliverable2_model_summary.csv",
    "results/tables/deliverable2_fold_metrics.csv",
    "results/tables/deliverable2_fold_safe_feature_audit.csv",
    "results/tables/deliverable2_research_feature_notes.csv",
    "results/tables/deliverable2_tsfresh_relevance_probe_summary.csv",
    "Deliverable2_FoldSafe_Implementation.ipynb",
    "results/figures/deliverable2_metric_summary.png",
    "results/figures/deliverable2_best_confusion_matrices.png",
  ], 70, 535, 860, 135, 12);
  addFooter(ctx, slide, n++, totalSlides);

  await fs.mkdir(path.dirname(OUT), { recursive: true });
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  for (let i = 0; i < presentation.slides.count; i++) {
    const s = presentation.slides.getItem(i);
    await saveBlobToFile(await presentation.export({ slide: s, format: "png", scale: 1 }), path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
    await fs.writeFile(path.join(LAYOUT_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await (await presentation.export({ slide: s, format: "layout" })).text(), "utf8");
  }
  console.log(JSON.stringify({ out: OUT, slides: presentation.slides.count, previewDir: PREVIEW_DIR }, null, 2));
}

await ensureArtifactToolWorkspace(WORKSPACE);
await build();
