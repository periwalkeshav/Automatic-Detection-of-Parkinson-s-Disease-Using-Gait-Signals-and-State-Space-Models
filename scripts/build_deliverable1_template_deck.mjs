#!/usr/bin/env node
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");

const __filename = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(__filename), "..");
const RESULTS = path.join(ROOT, "results");
const FIGURES = path.join(RESULTS, "figures");
const TABLES = path.join(RESULTS, "tables");
const TEMPLATE_ASSETS = path.join(RESULTS, "template_assets");
const OUT = path.join(RESULTS, "Deliverable_1_Gait_PD_Detection_Results.pptx");

const C = {
  navy: "204251",
  blue: "2F586E",
  steel: "8C9FB1",
  pale: "C2D0DC",
  light: "F3F7FA",
  paper: "FFFFFF",
  ink: "1B2A34",
  muted: "5F6F7C",
  line: "D9E2EA",
  patient: "B24545",
  control: "2F6F9F",
  good: "0D766E",
};

function readCsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const [header, ...rows] = text.split(/\r?\n/);
  const keys = header.split(",");
  return rows.map((row) => {
    const values = row.split(",");
    return Object.fromEntries(keys.map((key, i) => [key, values[i]]));
  });
}

function pct(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function addLogo(slide, dark = false) {
  const logo = path.join(TEMPLATE_ASSETS, dark ? "image2.png" : "image1.png");
  slide.addImage({ path: logo, x: 10.15, y: 0.28, w: 2.38, h: 0.83 });
}

function addFooter(slide, n) {
  slide.addShape("line", {
    x: 0.55,
    y: 7.02,
    w: 12.22,
    h: 0,
    line: { color: C.line, width: 0.75 },
  });
  slide.addText("May 2026 | Time Series Project | Deliverable 1", {
    x: 0.62,
    y: 7.12,
    w: 5.3,
    h: 0.18,
    fontFace: "Arial",
    fontSize: 7.5,
    color: C.muted,
    margin: 0,
  });
  slide.addText("Technische Fakultat", {
    x: 6.0,
    y: 7.12,
    w: 2.2,
    h: 0.18,
    fontFace: "Arial",
    fontSize: 7.5,
    color: C.muted,
    margin: 0,
  });
  slide.addText(String(n), {
    x: 12.24,
    y: 7.1,
    w: 0.45,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 8,
    bold: true,
    color: C.blue,
    align: "right",
    margin: 0,
  });
}

function addHeader(slide, title, subtitle, n) {
  slide.background = { color: C.paper };
  slide.addShape("rect", {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.13,
    fill: { color: C.blue },
    line: { color: C.blue, transparency: 100 },
  });
  slide.addText("Deliverable 1", {
    x: 0.65,
    y: 0.36,
    w: 1.8,
    h: 0.18,
    fontFace: "Arial",
    fontSize: 7.8,
    bold: true,
    color: C.steel,
    charSpace: 0,
    margin: 0,
  });
  slide.addText(title, {
    x: 0.62,
    y: 0.67,
    w: 8.6,
    h: 0.42,
    fontFace: "Arial",
    fontSize: 19,
    bold: true,
    color: C.navy,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.63,
      y: 1.11,
      w: 8.9,
      h: 0.24,
      fontFace: "Arial",
      fontSize: 9.3,
      color: C.muted,
      margin: 0,
    });
  }
  addLogo(slide, true);
  addFooter(slide, n);
}

function addSectionLabel(slide, text, x, y, color = C.blue) {
  slide.addText(text, {
    x,
    y,
    w: 3.0,
    h: 0.22,
    fontFace: "Arial",
    fontSize: 8.5,
    bold: true,
    color,
    margin: 0,
  });
}

function addKpi(slide, value, label, x, y, w = 1.74, color = C.blue) {
  slide.addShape("rect", {
    x,
    y,
    w,
    h: 0.9,
    fill: { color: C.light },
    line: { color: C.line, width: 0.7 },
    radius: 0.03,
  });
  slide.addText(value, {
    x: x + 0.14,
    y: y + 0.14,
    w: w - 0.28,
    h: 0.28,
    fontFace: "Arial",
    fontSize: 17.5,
    bold: true,
    color,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.14,
    y: y + 0.53,
    w: w - 0.28,
    h: 0.24,
    fontFace: "Arial",
    fontSize: 7.8,
    color: C.muted,
    fit: "shrink",
    margin: 0,
  });
}

function addBullets(slide, lines, x, y, w, h, size = 10.2) {
  slide.addText(lines.map((text) => ({ text, options: { bullet: { type: "bullet" } } })), {
    x,
    y,
    w,
    h,
    fontFace: "Arial",
    fontSize: size,
    color: C.ink,
    breakLine: false,
    fit: "shrink",
    paraSpaceAfterPt: 6,
    margin: 0,
  });
}

function addCallout(slide, title, body, x, y, w, h, color = C.blue) {
  slide.addShape("rect", {
    x,
    y,
    w,
    h,
    fill: { color: color, transparency: 5 },
    line: { color, width: 0.7 },
    radius: 0.04,
  });
  slide.addText(title, {
    x: x + 0.16,
    y: y + 0.16,
    w: w - 0.32,
    h: 0.22,
    fontFace: "Arial",
    fontSize: 9.5,
    bold: true,
    color: "FFFFFF",
    margin: 0,
  });
  slide.addText(body, {
    x: x + 0.16,
    y: y + 0.48,
    w: w - 0.32,
    h: h - 0.58,
    fontFace: "Arial",
    fontSize: 8.5,
    color: "FFFFFF",
    fit: "shrink",
    margin: 0,
  });
}

function addImageFrame(slide, file, x, y, w, h) {
  slide.addShape("rect", {
    x,
    y,
    w,
    h,
    fill: { color: "FFFFFF" },
    line: { color: C.line, width: 0.8 },
    radius: 0.03,
  });
  slide.addImage({ path: file, x: x + 0.05, y: y + 0.05, w: w - 0.1, h: h - 0.1 });
}

function addMetricBars(slide, rows, x, y, w) {
  const maxW = w - 2.8;
  rows.forEach((row, i) => {
    const yy = y + i * 0.42;
    const f1 = Number(row.test_f1_mean);
    slide.addText(`${row.scope} ${row.model.replace("_", " ")}`, {
      x,
      y: yy + 0.03,
      w: 2.4,
      h: 0.16,
      fontFace: "Arial",
      fontSize: 7.6,
      color: C.ink,
      margin: 0,
      fit: "shrink",
    });
    slide.addShape("rect", {
      x: x + 2.45,
      y: yy,
      w: maxW,
      h: 0.21,
      fill: { color: C.pale, transparency: 5 },
      line: { color: C.pale, transparency: 100 },
    });
    slide.addShape("rect", {
      x: x + 2.45,
      y: yy,
      w: maxW * f1,
      h: 0.21,
      fill: { color: i === 0 ? C.good : C.blue },
      line: { color: i === 0 ? C.good : C.blue, transparency: 100 },
    });
    slide.addText(pct(f1), {
      x: x + 2.52 + maxW,
      y: yy - 0.005,
      w: 0.6,
      h: 0.18,
      fontFace: "Arial",
      fontSize: 7.6,
      bold: i === 0,
      color: i === 0 ? C.good : C.ink,
      margin: 0,
      align: "right",
    });
  });
}

function main() {
  const demographics = readCsv(path.join(TABLES, "demographics_summary.csv"));
  const baseline = readCsv(path.join(TABLES, "baseline_summary.csv"));
  const signal = readCsv(path.join(TABLES, "subject_signal_summary.csv"));
  const combinedPca = readCsv(path.join(TABLES, "pca_combined_feet.csv"))[0];
  const leftPca = readCsv(path.join(TABLES, "pca_left_foot.csv"))[0];
  const rightPca = readCsv(path.join(TABLES, "pca_right_foot.csv"))[0];

  const control = demographics.find((row) => row.Group === "Control");
  const patient = demographics.find((row) => row.Group === "Patient");
  const topRows = [...baseline].sort((a, b) => Number(b.test_f1_mean) - Number(a.test_f1_mean));
  const best = topRows[0];
  const bestLinear = topRows.find((row) => row.model === "linear_svm");
  const durationMean = signal.reduce((acc, row) => acc + Number(row.used_duration_seconds), 0) / signal.length;

  const pptx = new pptxgen();
  pptx.author = "Codex";
  pptx.company = "Pattern Recognition Lab";
  pptx.subject = "Deliverable 1 results";
  pptx.title = "Gait-based Parkinson's Disease Detection";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Arial",
    bodyFontFace: "Arial",
    lang: "en-US",
  };
  pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
  pptx.layout = "CUSTOM_WIDE";

  let n = 1;

  let slide = pptx.addSlide();
  slide.addImage({ path: path.join(TEMPLATE_ASSETS, "image3.jpg"), x: 0, y: 0, w: 13.333, h: 7.5 });
  slide.addShape("rect", {
    x: 0,
    y: 0,
    w: 13.333,
    h: 7.5,
    fill: { color: C.navy, transparency: 21 },
    line: { color: C.navy, transparency: 100 },
  });
  slide.addShape("rect", {
    x: 0.72,
    y: 0.64,
    w: 3.05,
    h: 1.08,
    fill: { color: "FFFFFF", transparency: 4 },
    line: { color: "FFFFFF", transparency: 100 },
    radius: 0.04,
  });
  slide.addImage({ path: path.join(TEMPLATE_ASSETS, "image2.png"), x: 0.88, y: 0.79, w: 2.55, h: 0.89 });
  slide.addText("Time Series Project SS26 | Deliverable 1", {
    x: 0.78,
    y: 2.08,
    w: 5.2,
    h: 0.24,
    fontFace: "Arial",
    fontSize: 10.5,
    bold: true,
    color: C.pale,
    margin: 0,
  });
  slide.addText("Gait-Based Parkinson's Disease Detection", {
    x: 0.76,
    y: 2.52,
    w: 10.5,
    h: 0.82,
    fontFace: "Arial",
    fontSize: 32,
    bold: true,
    color: "FFFFFF",
    margin: 0,
    fit: "shrink",
  });
  slide.addText("Data analysis, feature engineering, and non-Mamba baseline results using the provided 5-fold protocol.", {
    x: 0.8,
    y: 3.5,
    w: 8.7,
    h: 0.35,
    fontFace: "Arial",
    fontSize: 13,
    color: "FFFFFF",
    margin: 0,
    fit: "shrink",
  });
  addKpi(slide, "100", "walk-01 subjects", 0.82, 5.35, 1.55, "FFFFFF");
  addKpi(slide, pct(best.test_accuracy_mean), "best test accuracy", 2.62, 5.35, 1.8, "FFFFFF");
  addKpi(slide, pct(best.test_f1_mean), "best test F1", 4.68, 5.35, 1.55, "FFFFFF");
  slide.addText("May 2026", {
    x: 0.82,
    y: 6.65,
    w: 2.2,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 8.5,
    color: C.pale,
    margin: 0,
  });

  slide = pptx.addSlide();
  addHeader(slide, "Executive Summary", "Non-Mamba baselines now exceed the reference linear-SVM range while preserving the official split protocol.", n++);
  addCallout(
    slide,
    "Main result",
    `Enhanced features with sparse L1 logistic regression reach ${pct(best.test_accuracy_mean)} mean test accuracy and ${pct(best.test_f1_mean)} mean test F1 across the five folds.`,
    0.72,
    1.65,
    4.05,
    1.25,
    C.good,
  );
  addKpi(slide, "50 / 50", "controls / patients", 5.18, 1.65, 1.72, C.blue);
  addKpi(slide, "5", "predefined folds", 7.12, 1.65, 1.5, C.blue);
  addKpi(slide, "60 / 20 / 20", "train / validation / test per fold", 8.84, 1.65, 2.35, C.blue);
  addBullets(
    slide,
    [
      "The data is balanced by class in every training, validation, and test split.",
      "Model selection uses validation data only; final metrics are averaged over held-out fold test sets.",
      "The strongest linear SVM reaches 79.0% mean test accuracy, above the 73% reference mentioned in the meeting slides.",
      "The best overall non-Mamba model is sparse logistic regression on enhanced features.",
    ],
    0.9,
    3.52,
    5.45,
    1.6,
  );
  addMetricBars(slide, topRows.slice(0, 7), 6.65, 3.5, 5.7);

  slide = pptx.addSlide();
  addHeader(slide, "Dataset and Fold Protocol", "The analysis uses only the provided walk-01 files and keeps testing isolated by fold.", n++);
  addKpi(slide, control.subjects, "control subjects", 0.72, 1.62, 1.65, C.control);
  addKpi(slide, patient.subjects, "PD subjects", 2.62, 1.62, 1.65, C.patient);
  addKpi(slide, `${control.age_mean} / ${patient.age_mean}`, "mean age: control / PD", 4.52, 1.62, 2.1, C.blue);
  addKpi(slide, "25 / 25", "male / female in each class", 6.88, 1.62, 2.15, C.blue);
  addKpi(slide, `${durationMean.toFixed(1)}s`, "mean analyzed duration", 9.28, 1.62, 1.75, C.blue);
  addSectionLabel(slide, "Fold structure", 0.8, 3.12);
  const foldRows = [
    ["Split", "Controls", "PD patients", "Total"],
    ["Training", "30", "30", "60"],
    ["Validation", "10", "10", "20"],
    ["Test", "10", "10", "20"],
  ];
  slide.addTable(foldRows, {
    x: 0.78,
    y: 3.48,
    w: 5.15,
    h: 1.55,
    colW: [1.55, 1.1, 1.35, 1.05],
    rowH: [0.34, 0.38, 0.38, 0.38],
    fontFace: "Arial",
    fontSize: 9,
    color: C.ink,
    border: { type: "solid", color: C.line, pt: 0.6 },
    fill: "FFFFFF",
    margin: 0.05,
    autoFit: false,
  });
  addBullets(
    slide,
    [
      "Each fold repeats the same class balance, which removes the need for SMOTE in this first pass.",
      "The validation split controls model and feature-count selection.",
      "The test split remains untouched until final scoring.",
    ],
    6.55,
    3.43,
    5.0,
    1.35,
  );

  slide = pptx.addSlide();
  addHeader(slide, "Feature Engineering Pipeline", "Literature and repository review motivated variability, asymmetry, and time-frequency descriptors.", n++);
  const steps = [
    ["Trim", "Remove 5s from start and end"],
    ["Window", "5s windows, 2.5s step"],
    ["TSFresh", "Curated force-signal features"],
    ["Engineer", "Timing, COP, asymmetry, wavelets"],
    ["Select", "Fold-safe SelectKBest"],
    ["Model", "RF, ExtraTrees, SVM, sparse logistic"],
  ];
  steps.forEach((step, i) => {
    const x = 0.65 + i * 2.08;
    slide.addShape("rect", {
      x,
      y: 1.74,
      w: 1.55,
      h: 1.02,
      fill: { color: i % 2 === 0 ? C.light : "FFFFFF" },
      line: { color: C.line, width: 0.7 },
      radius: 0.03,
    });
    slide.addText(step[0], {
      x: x + 0.12,
      y: 1.94,
      w: 1.32,
      h: 0.2,
      fontFace: "Arial",
      fontSize: 9.8,
      bold: true,
      color: C.navy,
      margin: 0,
    });
    slide.addText(step[1], {
      x: x + 0.12,
      y: 2.25,
      w: 1.32,
      h: 0.32,
      fontFace: "Arial",
      fontSize: 7.2,
      color: C.muted,
      fit: "shrink",
      margin: 0,
    });
    if (i < steps.length - 1) {
      slide.addShape("line", { x: x + 1.62, y: 2.25, w: 0.34, h: 0, line: { color: C.blue, width: 1.1, endArrowType: "triangle" } });
    }
  });
  addCallout(
    slide,
    "Enhanced representation",
    "Left and right TSFresh vectors are combined with paired absolute differences, gait contact statistics, force symmetry, center-of-pressure descriptors, and Morlet wavelet power summaries.",
    0.78,
    3.45,
    5.35,
    1.55,
    C.blue,
  );
  addBullets(
    slide,
    [
      "Gait literature emphasizes rhythmicity, variability, asymmetry, stance timing, and double support.",
      "The GitHub CNN/scalogram work suggested capturing time-frequency information; here it is added as classical wavelet features.",
      "All scaling, imputation, feature selection, and model selection are performed inside the fold pipeline.",
    ],
    6.65,
    3.45,
    5.35,
    1.55,
  );

  slide = pptx.addSlide();
  addHeader(slide, "Representative Force Signals", "Total vertical ground-reaction force over the first 10 seconds after trimming.", n++);
  addImageFrame(slide, path.join(FIGURES, "example_left_total_force.png"), 0.66, 1.45, 5.9, 3.42);
  addImageFrame(slide, path.join(FIGURES, "example_right_total_force.png"), 6.78, 1.45, 5.9, 3.42);
  addBullets(
    slide,
    [
      "The patient/control examples show differences in rhythm and force timing.",
      "Single traces are illustrative only; classification uses subject-level summaries across the full recording.",
    ],
    0.88,
    5.35,
    10.7,
    0.55,
    9.2,
  );

  slide = pptx.addSlide();
  addHeader(slide, "PCA Diagnostics", "The first two principal components help visualize structure but do not cleanly separate the classes.", n++);
  addImageFrame(slide, path.join(FIGURES, "pca_left_foot.png"), 0.58, 1.48, 3.85, 3.05);
  addImageFrame(slide, path.join(FIGURES, "pca_right_foot.png"), 4.75, 1.48, 3.85, 3.05);
  addImageFrame(slide, path.join(FIGURES, "pca_combined_feet.png"), 8.92, 1.48, 3.85, 3.05);
  slide.addText(`Left PC1+PC2: ${((Number(leftPca.pc1_explained) + Number(leftPca.pc2_explained)) * 100).toFixed(1)}% variance`, {
    x: 0.72,
    y: 4.75,
    w: 3.45,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 8.2,
    color: C.muted,
    margin: 0,
    align: "center",
  });
  slide.addText(`Right PC1+PC2: ${((Number(rightPca.pc1_explained) + Number(rightPca.pc2_explained)) * 100).toFixed(1)}% variance`, {
    x: 4.9,
    y: 4.75,
    w: 3.45,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 8.2,
    color: C.muted,
    margin: 0,
    align: "center",
  });
  slide.addText(`Combined PC1+PC2: ${((Number(combinedPca.pc1_explained) + Number(combinedPca.pc2_explained)) * 100).toFixed(1)}% variance`, {
    x: 9.08,
    y: 4.75,
    w: 3.45,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 8.2,
    color: C.muted,
    margin: 0,
    align: "center",
  });
  addBullets(
    slide,
    [
      "PCA is useful as a sanity check, not as the final separation mechanism.",
      "Supervised feature selection and nonlinear/regularized baselines capture more discriminative structure.",
    ],
    0.9,
    5.38,
    10.8,
    0.55,
    9,
  );

  slide = pptx.addSlide();
  addHeader(slide, "Baseline Results", "The enhanced feature set gives the best fold-averaged test performance before Mamba/SSM models.", n++);
  const displayRows = [
    ["Scope", "Model", "Val F1", "Test acc.", "Test F1"],
    ...topRows.slice(0, 9).map((row) => [
      row.scope,
      row.model.replace("_", " "),
      pct(row.validation_f1_mean),
      pct(row.test_accuracy_mean),
      pct(row.test_f1_mean),
    ]),
  ];
  slide.addTable(displayRows, {
    x: 0.72,
    y: 1.52,
    w: 6.72,
    h: 3.88,
    colW: [1.25, 1.55, 1.15, 1.35, 1.2],
    rowH: Array(displayRows.length).fill(0.34),
    fontFace: "Arial",
    fontSize: 8.1,
    color: C.ink,
    border: { type: "solid", color: C.line, pt: 0.55 },
    fill: "FFFFFF",
    margin: 0.05,
    autoFit: false,
    fit: "shrink",
  });
  slide.addShape("rect", {
    x: 0.72,
    y: 1.52,
    w: 6.72,
    h: 0.34,
    fill: { color: C.blue },
    line: { color: C.blue, transparency: 100 },
  });
  addCallout(
    slide,
    "Best model",
    `${best.scope} ${best.model.replace("_", " ")}: ${pct(best.test_accuracy_mean)} mean test accuracy and ${pct(best.test_f1_mean)} mean test F1.`,
    8.0,
    1.62,
    3.82,
    1.12,
    C.good,
  );
  addCallout(
    slide,
    "Linear SVM reference",
    `Enhanced linear SVM reaches ${pct(bestLinear.test_accuracy_mean)} mean test accuracy, exceeding the 73% meeting-slide reference.`,
    8.0,
    3.02,
    3.82,
    1.12,
    C.blue,
  );
  addBullets(
    slide,
    [
      "All rows use the same five train/validation/test folds.",
      "The enhanced scope includes TSFresh, asymmetry, gait, COP, and wavelet features.",
      "L1 logistic is attractive because it is sparse and easier to explain than deeper models.",
    ],
    8.02,
    4.55,
    3.82,
    0.95,
    8.7,
  );

  slide = pptx.addSlide();
  addHeader(slide, "What Improved the Model", "The gain came from adding domain-informed descriptors while keeping the validation protocol strict.", n++);
  const cards = [
    ["Gait timing", "Contact duration, swing intervals, cadence, double support, and step interval variability."],
    ["Left-right asymmetry", "Paired absolute differences between left and right TSFresh features plus force symmetry descriptors."],
    ["Time-frequency power", "Morlet wavelet power captures rhythmic energy patterns without training a CNN yet."],
    ["Sparse selection", "L1 logistic and SelectKBest reduce high-dimensional noise on only 100 subjects."],
  ];
  cards.forEach((card, i) => {
    const x = 0.78 + (i % 2) * 6.05;
    const y = 1.55 + Math.floor(i / 2) * 1.65;
    slide.addShape("rect", {
      x,
      y,
      w: 5.35,
      h: 1.12,
      fill: { color: i === 0 ? C.light : "FFFFFF" },
      line: { color: C.line, width: 0.75 },
      radius: 0.04,
    });
    slide.addText(card[0], {
      x: x + 0.18,
      y: y + 0.18,
      w: 4.85,
      h: 0.22,
      fontFace: "Arial",
      fontSize: 11,
      bold: true,
      color: C.navy,
      margin: 0,
    });
    slide.addText(card[1], {
      x: x + 0.18,
      y: y + 0.5,
      w: 4.85,
      h: 0.42,
      fontFace: "Arial",
      fontSize: 8.4,
      color: C.muted,
      fit: "shrink",
      margin: 0,
    });
  });
  addCallout(
    slide,
    "Important caveat",
    "Published papers often report higher scores under different protocols, full datasets, balancing strategies, or cross-validation choices. These results use the provided folds directly.",
    0.82,
    5.15,
    11.1,
    0.92,
    C.blue,
  );

  slide = pptx.addSlide();
  addHeader(slide, "Conclusion and Next Steps", "Deliverable 1 is complete as a reproducible non-Mamba baseline package.", n++);
  addCallout(
    slide,
    "Current best",
    `${pct(best.test_accuracy_mean)} accuracy / ${pct(best.test_f1_mean)} F1 with enhanced sparse logistic regression.`,
    0.78,
    1.55,
    4.25,
    1.05,
    C.good,
  );
  addBullets(
    slide,
    [
      "The project now has fold manifests, demographics summaries, signal diagnostics, PCA plots, feature matrices, and baseline metrics.",
      "The strongest baseline is interpretable and improves meaningfully over the first TSFresh-only result.",
      "Mamba/SSM modeling can now start from a stronger and better documented classical baseline.",
    ],
    0.9,
    3.05,
    5.45,
    1.4,
  );
  addSectionLabel(slide, "Suggested next experiments", 7.1, 1.6, C.good);
  addBullets(
    slide,
    [
      "Inspect selected L1 features per fold to identify stable gait markers.",
      "Add fold-safe probability calibration and confusion matrices.",
      "Then implement Mamba/SSM and compare against this enhanced baseline.",
    ],
    7.12,
    2.05,
    4.65,
    1.35,
  );
  slide.addImage({ path: path.join(TEMPLATE_ASSETS, "image4.jpeg"), x: 7.12, y: 4.05, w: 4.65, h: 1.74 });

  pptx.writeFile({ fileName: OUT });
  console.log(`Wrote ${OUT}`);
}

main();
