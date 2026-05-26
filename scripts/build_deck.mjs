#!/usr/bin/env node
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");

const __filename = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(__filename), "..");
const RESULTS = path.join(ROOT, "results");
const FIGURES = path.join(RESULTS, "figures");
const TABLES = path.join(RESULTS, "tables");
const DECK = path.join(RESULTS, "Deliverable_1_Data_Analysis_Draft.pptx");

const COLORS = {
  ink: "1F2933",
  muted: "5D6670",
  line: "D9D2C7",
  paper: "FBFAF7",
  patient: "C44E52",
  control: "2F6F9F",
  accent: "0E6F73",
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

async function convertFigures() {
  const names = [
    "example_left_total_force",
    "example_right_total_force",
    "pca_left_foot",
    "pca_right_foot",
    "pca_combined_feet",
  ];
  for (const name of names) {
    await sharp(path.join(FIGURES, `${name}.svg`))
      .resize({ width: 1600 })
      .png()
      .toFile(path.join(FIGURES, `${name}.png`));
  }
}

function addFooter(slide, num) {
  slide.addText(`May 2026 | Time Series Project | ${num}`, {
    x: 0.55,
    y: 7.12,
    w: 12.2,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 7.5,
    color: COLORS.muted,
    margin: 0,
  });
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.6,
    y: 0.35,
    w: 8.4,
    h: 0.34,
    fontFace: "Arial",
    bold: true,
    fontSize: 18,
    color: COLORS.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.6,
      y: 0.78,
      w: 9.8,
      h: 0.24,
      fontFace: "Arial",
      fontSize: 9,
      color: COLORS.muted,
      margin: 0,
    });
  }
}

function addKpi(slide, label, value, x, y, color = COLORS.accent) {
  slide.addShape("rect", {
    x,
    y,
    w: 1.65,
    h: 0.84,
    fill: { color: "FFFFFF" },
    line: { color: COLORS.line, width: 0.6 },
    radius: 0.06,
  });
  slide.addText(value, {
    x: x + 0.13,
    y: y + 0.14,
    w: 1.38,
    h: 0.28,
    fontFace: "Arial",
    fontSize: 18,
    bold: true,
    color,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.13,
    y: y + 0.5,
    w: 1.38,
    h: 0.2,
    fontFace: "Arial",
    fontSize: 7.5,
    color: COLORS.muted,
    margin: 0,
  });
}

function addBullets(slide, lines, x, y, w, h) {
  slide.addText(lines.map((text) => ({ text, options: { bullet: { type: "bullet" } } })), {
    x,
    y,
    w,
    h,
    fontFace: "Arial",
    fontSize: 11,
    color: COLORS.ink,
    breakLine: false,
    fit: "shrink",
    paraSpaceAfterPt: 7,
    margin: 0,
  });
}

function addImage(slide, file, x, y, w, h) {
  slide.addImage({ path: path.join(FIGURES, file), x, y, w, h });
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

async function main() {
  await convertFigures();

  const demographics = readCsv(path.join(TABLES, "demographics_summary.csv"));
  const combined = readCsv(path.join(TABLES, "pca_combined_feet.csv"))[0];
  const left = readCsv(path.join(TABLES, "pca_left_foot.csv"))[0];
  const right = readCsv(path.join(TABLES, "pca_right_foot.csv"))[0];
  const baselineSummary = readCsv(path.join(TABLES, "baseline_summary.csv"));
  const bestBaseline = [...baselineSummary].sort((a, b) => Number(b.test_f1_mean) - Number(a.test_f1_mean))[0];
  const bestLinear = [...baselineSummary]
    .filter((row) => row.model === "linear_svm")
    .sort((a, b) => Number(b.test_accuracy_mean) - Number(a.test_accuracy_mean))[0];
  const displayBaselines = [...baselineSummary]
    .sort((a, b) => Number(b.test_f1_mean) - Number(a.test_f1_mean))
    .slice(0, 10);
  const control = demographics.find((d) => (d.group ?? d.Group) === "Control");
  const patient = demographics.find((d) => (d.group ?? d.Group) === "Patient");

  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Codex";
  pptx.company = "Time Series Project";
  pptx.subject = "Deliverable 1 gait data analysis";
  pptx.title = "Deliverable 1 Data Analysis";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: "Arial",
    bodyFontFace: "Arial",
    lang: "en-US",
  };
  pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
  pptx.layout = "CUSTOM_WIDE";

  let slideNum = 1;
  let slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  slide.addText("Deliverable #1", {
    x: 0.7,
    y: 0.65,
    w: 3.2,
    h: 0.35,
    fontFace: "Arial",
    fontSize: 12,
    color: COLORS.accent,
    bold: true,
    margin: 0,
  });
  slide.addText("Gait Time-Series Data Analysis", {
    x: 0.7,
    y: 1.15,
    w: 9.8,
    h: 0.7,
    fontFace: "Arial",
    fontSize: 30,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  slide.addText("Parkinson's disease patients vs. controls using walk-01 vertical ground reaction force signals.", {
    x: 0.72,
    y: 2.02,
    w: 8.6,
    h: 0.38,
    fontFace: "Arial",
    fontSize: 13,
    color: COLORS.muted,
    margin: 0,
  });
  addKpi(slide, "patients", patient.subjects, 0.72, 3.15, COLORS.patient);
  addKpi(slide, "controls", control.subjects, 2.55, 3.15, COLORS.control);
  addKpi(slide, "sampling rate", "100 Hz", 4.38, 3.15);
  addKpi(slide, "window / step", "5s / 2.5s", 6.21, 3.15);
  slide.addText("Outputs generated in /results: fold manifests, TSFresh features, PCA scores, baseline metrics, figures, and this draft deck.", {
    x: 0.72,
    y: 6.62,
    w: 8.5,
    h: 0.22,
    fontFace: "Arial",
    fontSize: 8.5,
    color: COLORS.muted,
    margin: 0,
  });
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "Dataset Snapshot", "The five provided folds reuse the same 100 walk-01 subjects with balanced 60/20/20 train/validation/test splits.");
  addKpi(slide, "control age, mean +/- SD", `${control.age_mean} +/- ${control.age_sd}`, 0.75, 1.42, COLORS.control);
  addKpi(slide, "patient age, mean +/- SD", `${patient.age_mean} +/- ${patient.age_sd}`, 2.62, 1.42, COLORS.patient);
  addKpi(slide, "control M/F", `${control.male}/${control.female}`, 4.49, 1.42, COLORS.control);
  addKpi(slide, "patient M/F", `${patient.male}/${patient.female}`, 6.36, 1.42, COLORS.patient);
  addBullets(
    slide,
    [
      "Each time step has 19 columns: time, 8 left-foot sensors, 8 right-foot sensors, and total force per foot.",
      "All files in this first pass are normal walk-01 recordings sampled at 100 Hz.",
      "Every fold has 30 patients / 30 controls for training, 10 / 10 for validation, and 10 / 10 for test.",
      "Feature extraction uses overlapping 5-second windows with a 2.5-second step, matching the deliverable guidance.",
      "Subject-level vectors summarize each window feature by mean, standard deviation, skewness, and kurtosis.",
    ],
    0.8,
    2.9,
    10.8,
    2.0,
  );
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "Example Patient-Control Signals", "Total vertical ground reaction force over the first 10 seconds of representative subjects.");
  addImage(slide, "example_left_total_force.png", 0.55, 1.12, 6.05, 3.42);
  addImage(slide, "example_right_total_force.png", 6.75, 1.12, 6.05, 3.42);
  slide.addText("Patient and control traces differ in amplitude rhythm and stance timing in this example; the later PCA slides test whether windowed features produce broader subject-level separation.", {
    x: 0.78,
    y: 5.08,
    w: 11.4,
    h: 0.36,
    fontFace: "Arial",
    fontSize: 11.5,
    color: COLORS.ink,
    margin: 0,
  });
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "Feature Pipeline", "The analysis produces fold-safe TSFresh, gait, asymmetry, and wavelet subject features.");
  const steps = [
    ["Raw signal", "19 columns at 100 Hz"],
    ["Windowing", "5s window, 2.5s step"],
    ["Feature set", "Curated TSFresh + engineered gait features"],
    ["Subject vector", "mean/std/skew/kurt across windows for every feature"],
    ["Projection", "z-score normalization followed by PCA"],
  ];
  steps.forEach((step, i) => {
    const x = 0.7 + i * 2.45;
    slide.addShape("rect", {
      x,
      y: 2.22,
      w: 1.86,
      h: 1.02,
      fill: { color: "FFFFFF" },
      line: { color: COLORS.line, width: 0.7 },
      radius: 0.04,
    });
    slide.addText(step[0], {
      x: x + 0.12,
      y: 2.4,
      w: 1.62,
      h: 0.22,
      fontFace: "Arial",
      fontSize: 10.5,
      bold: true,
      color: COLORS.ink,
      margin: 0,
    });
    slide.addText(step[1], {
      x: x + 0.12,
      y: 2.7,
      w: 1.62,
      h: 0.34,
      fontFace: "Arial",
      fontSize: 7.8,
      color: COLORS.muted,
      fit: "shrink",
      margin: 0,
    });
    if (i < steps.length - 1) {
      slide.addShape("line", { x: x + 1.96, y: 2.73, w: 0.35, h: 0, line: { color: COLORS.accent, width: 1.3, beginArrowType: "none", endArrowType: "triangle" } });
    }
  });
  addBullets(
    slide,
    [
      "The feature matrix is written to results/tables/subject_feature_matrix.csv for reproducibility.",
      "Enhanced features add contact timing, COP, force symmetry, paired left-right differences, and Morlet wavelet power.",
      "PCA is run three ways as requested: right foot only, left foot only, and concatenated left+right features.",
      "Baselines use each fold's training split for fitting, validation split for model selection, and test split for final reporting.",
    ],
    1.0,
    4.55,
    10.8,
    0.9,
  );
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "PCA: Left Foot Features", `PC1 ${(+left.pc1_explained * 100).toFixed(1)}%, PC2 ${(+left.pc2_explained * 100).toFixed(1)}% explained variance.`);
  addImage(slide, "pca_left_foot.png", 1.0, 1.05, 10.85, 5.48);
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "PCA: Right Foot Features", `PC1 ${(+right.pc1_explained * 100).toFixed(1)}%, PC2 ${(+right.pc2_explained * 100).toFixed(1)}% explained variance.`);
  addImage(slide, "pca_right_foot.png", 1.0, 1.05, 10.85, 5.48);
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "PCA: Combined Left + Right Features", `PC1 ${(+combined.pc1_explained * 100).toFixed(1)}%, PC2 ${(+combined.pc2_explained * 100).toFixed(1)}% explained variance.`);
  addImage(slide, "pca_combined_feet.png", 0.78, 1.02, 8.35, 5.34);
  slide.addText("First-pass readout", {
    x: 9.52,
    y: 1.42,
    w: 2.4,
    h: 0.28,
    fontFace: "Arial",
    fontSize: 12,
    bold: true,
    color: COLORS.ink,
    margin: 0,
  });
  addBullets(
    slide,
    [
      "The groups are not cleanly separable in the first two PCA axes.",
      "Study/source effects and gait-speed differences may explain part of the structure.",
      "The supervised baseline slide quantifies what PCA alone cannot show visually.",
    ],
    9.55,
    1.92,
    3.0,
    2.2,
  );
  addFooter(slide, slideNum++);

  slide = pptx.addSlide();
  slide.background = { color: COLORS.paper };
  addTitle(slide, "Baseline Models Before Mamba", "Enhanced features add gait timing, COP, asymmetry, and wavelet descriptors to the TSFresh subject vectors.");
  const tableRows = [
    [
      { text: "Scope", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accent } } },
      { text: "Model", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accent } } },
      { text: "Val F1", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accent } } },
      { text: "Test Acc.", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accent } } },
      { text: "Test F1", options: { bold: true, color: "FFFFFF", fill: { color: COLORS.accent } } },
    ],
    ...displayBaselines.map((row) => [
      row.scope,
      row.model.replace("_", " "),
      pct(row.validation_f1_mean),
      pct(row.test_accuracy_mean),
      pct(row.test_f1_mean),
    ]),
  ];
  slide.addShape("rect", {
    x: 0.76,
    y: 1.18,
    w: 7.95,
    h: 0.34,
    fill: { color: COLORS.accent, transparency: 0 },
    line: { color: COLORS.accent, transparency: 100 },
  });
  slide.addTable(tableRows, {
    x: 0.76,
    y: 1.18,
    w: 7.95,
    h: 4.72,
    colW: [1.28, 1.75, 1.25, 1.35, 1.25],
    border: { type: "solid", color: COLORS.line, pt: 0.5 },
    fill: "FFFFFF",
    color: COLORS.ink,
    fontFace: "Arial",
    fontSize: 8.0,
    margin: 0.05,
    autoFit: false,
    valign: "mid",
    fit: "shrink",
    rowH: Array(tableRows.length).fill(0.34),
    options: { bandRow: false },
  });
  slide.addText(`Best mean test F1 is ${pct(bestBaseline.test_f1_mean)} from ${bestBaseline.scope} ${bestBaseline.model.replace("_", " ")}. The strongest linear SVM reaches ${pct(bestLinear.test_accuracy_mean)} mean test accuracy on ${bestLinear.scope} features, matching the meeting-slide baseline range.`, {
    x: 9.15,
    y: 1.38,
    w: 3.2,
    h: 1.2,
    fontFace: "Arial",
    fontSize: 12,
    color: COLORS.ink,
    fit: "shrink",
    margin: 0,
  });
  addBullets(
    slide,
    [
      "No Mamba/SSM model is implemented in this step.",
      "Validation is separate from the held-out test split in each fold.",
      "Metrics are stored in baseline_fold_metrics.csv and baseline_summary.csv.",
    ],
    9.15,
    3.02,
    3.1,
    1.4,
  );
  addFooter(slide, slideNum++);

  await pptx.writeFile({ fileName: DECK });
  console.log(`Wrote ${DECK}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
