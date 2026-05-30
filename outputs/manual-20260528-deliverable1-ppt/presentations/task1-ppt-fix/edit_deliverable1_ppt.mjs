import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "/Users/keshavperiwal/Documents Local/Semester 2/Project Time Series/Data";
const sourcePptx = path.join(root, "Deliverable1_Presentation.pptx");
const outputDir = path.join(root, "results");
const finalPptx = path.join(outputDir, "Deliverable1_Presentation_Task1_fixed.pptx");
const previewDir = path.join(root, "outputs/manual-20260528-deliverable1-ppt/presentations/task1-ppt-fix/final-preview");
const layoutDir = path.join(root, "outputs/manual-20260528-deliverable1-ppt/presentations/task1-ppt-fix/final-layout");

const img = (name) => path.join(root, "deliverable1_outputs", name);

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function saveBlob(blob, filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

function slide(presentation, number) {
  return presentation.slides.getItem(number - 1);
}

function shape(slideObject, id) {
  return slideObject.shapes.getById(String(id));
}

function setText(slideObject, id, text) {
  const target = shape(slideObject, id);
  target.text.set(text);
  return target;
}

function deleteShape(slideObject, id) {
  try {
    shape(slideObject, id).delete();
  } catch {
    // Some imported template elements are absent after PPTX normalization.
  }
}

async function addImage(slideObject, imagePath, frame, fit = "contain", alt = "") {
  const image = slideObject.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt: alt || path.basename(imagePath),
    name: path.basename(imagePath),
  });
  image.position = frame;
  return image;
}

async function replaceShapeWithImage(slideObject, id, imagePath, frameOverride, fit = "contain") {
  const target = shape(slideObject, id);
  const frame = frameOverride || target.frame;
  target.delete();
  return addImage(slideObject, imagePath, frame, fit);
}

function removeAgendaTemplateNoise(slideObject) {
  for (const id of [3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 30]) {
    deleteShape(slideObject, id);
  }
}

function updateVisibleFooters(presentation) {
  const total = presentation.slides.count;
  for (let index = 0; index < total; index += 1) {
    const s = presentation.slides.getItem(index);
    for (const item of s.shapes.items) {
      const current = String(item.text || "").trim();
      if (/^\d+\/\d+$/.test(current)) {
        item.text.set(`${index + 1}/${total}`);
      }
      if (/^Slide\s+\d+/i.test(current)) {
        item.text.set(`${index + 1}/${total}`);
      }
    }
  }
}

function removeEmptyStructuralPlaceholders(presentation) {
  for (let index = 0; index < presentation.slides.count; index += 1) {
    const s = presentation.slides.getItem(index);
    for (const item of [...s.shapes.items]) {
      const isEmptyText = String(item.text || "").trim().length === 0;
      if (item.isPlaceholder && isEmptyText) {
        item.delete();
      }
    }
  }
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));

// Agenda cleanup and copy update.
{
  const s = slide(presentation, 2);
  removeAgendaTemplateNoise(s);
  setText(s, 33, "Dataset Overview");
  setText(s, 34, "Full demographics and fold split structure");
  setText(s, 39, "Windowed TSFresh/manual features, z-score normalization");
  setText(s, 44, "05");
  setText(s, 45, "Feature Diagnostics");
  setText(s, 46, "Heatmap and combined-feature distributions");
  setText(s, 47, "06");
  setText(s, 48, "Key Findings");
  setText(s, 49, "Task 1 observations and presentation takeaways");
  setText(s, 50, "Agenda - Deliverable 1: Data Analysis");
}

// Full-demographics figure.
{
  const s = slide(presentation, 4);
  deleteShape(s, 6);
  await replaceShapeWithImage(
    s,
    55,
    img("full_demographics_distribution.png"),
    { left: 270, top: 474, width: 740, height: 185 },
    "contain",
  );
}

// Fold/split slide: include validation and all 5-fold split structure.
{
  const s = slide(presentation, 5);
  setText(s, 5, "Dataset Demographics - 5-Fold Split Files");
  deleteShape(s, 6);
  setText(s, 8, "Per-fold composition (validation included)");

  const rows = [
    ["Training files / fold", "30", "30", "60"],
    ["Validation files / fold", "10", "10", "20"],
    ["Test files / fold", "10", "10", "20"],
    ["Per-fold total", "50", "50", "100"],
    ["Unique subject set", "50", "50", "100"],
    ["Cohorts Ga / Si / Ju", "18 / 16 / 16", "13 / 21 / 16", "31 / 37 / 32"],
    ["PD H&Y 2.0 / 2.5 / 3.0", "26 / 16 / 8", "-", "50 PD"],
  ];
  const rowShapeIds = [
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31, 32],
    [33, 34, 35, 36],
    [37, 38, 39, 40],
  ];
  rows.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => setText(s, rowShapeIds[rowIndex][colIndex], value));
  });
  setText(s, 41, "Split Integrity Note");
  setText(
    s,
    42,
    [
      "- Every fold is class-balanced: 30/30 train, 10/10 validation, 10/10 test.",
      "- The same 100-subject balanced subset appears across the five split files.",
      "- Validation should be used for model selection; test remains held out.",
      "- Prepared feature figures in deliverable1_outputs are from the existing Fold 1 train+test output subset (N=80).",
    ].join("\n"),
  );
  await replaceShapeWithImage(
    s,
    43,
    img("fold_subject_distribution.png"),
    { left: 653, top: 362, width: 566, height: 142 },
    "contain",
  );
}

// Signal and embedding placeholders.
await replaceShapeWithImage(slide(presentation, 7), 9, img("fig1_raw_signals.png"), { left: 28.8, top: 198, width: 1219.2, height: 455 }, "contain");
await replaceShapeWithImage(slide(presentation, 8), 9, img("fig2_variability_comparison.png"), { left: 28.8, top: 205, width: 1219.2, height: 448 }, "contain");
await replaceShapeWithImage(slide(presentation, 12), 9, img("pca_left_foot.png"), { left: 28.8, top: 205, width: 1219.2, height: 336 }, "contain");
await replaceShapeWithImage(slide(presentation, 13), 9, img("pca_right_foot.png"), { left: 28.8, top: 205, width: 1219.2, height: 336 }, "contain");
await replaceShapeWithImage(slide(presentation, 14), 9, img("pca_combined.png"), { left: 28.8, top: 205, width: 1219.2, height: 336 }, "contain");

await replaceShapeWithImage(slide(presentation, 15), 9, img("tsne_left_foot.png"), { left: 19.2, top: 205, width: 408, height: 170 }, "contain");
await replaceShapeWithImage(slide(presentation, 15), 11, img("tsne_right_foot.png"), { left: 441.6, top: 205, width: 408, height: 170 }, "contain");
await replaceShapeWithImage(slide(presentation, 15), 13, img("tsne_combined.png"), { left: 864, top: 205, width: 408, height: 170 }, "contain");

// Feature diagnostics section and slides. Repurpose the old UMAP placeholder slide to avoid unsupported placeholders.
{
  const s = slide(presentation, 16);
  setText(s, 4, "05");
  setText(s, 5, "Feature Diagnostics");
  setText(s, 6, "Heatmaps - Feature Distributions - PCA/t-SNE Interpretation");
}

{
  const s = slide(presentation, 17);
  setText(s, 5, "Feature Analysis - Heatmap");
  setText(s, 6, "05 - Feature Diagnostics");
  setText(s, 7, "Top discriminative features by foot condition");
  setText(s, 8, "Group-wise z-score means highlight which extracted features differ between PD and healthy controls.");
  deleteShape(s, 9);
  deleteShape(s, 10);
  deleteShape(s, 11);
  deleteShape(s, 12);
  await addImage(s, img("feature_heatmap.png"), { left: 38, top: 225, width: 1204, height: 425 }, "contain");
}

{
  const s = slide(presentation, 18);
  setText(s, 5, "Feature Distributions - Combined Feet");
  setText(s, 6, "05 - Feature Diagnostics");
  setText(s, 7, "KDE distributions for selected combined-foot features");
  deleteShape(s, 8);
  deleteShape(s, 9);
  deleteShape(s, 10);
  deleteShape(s, 11);
  deleteShape(s, 12);
  await addImage(s, img("feature_distributions_combined.png"), { left: 198, top: 190, width: 884, height: 468 }, "contain");
}

// Summary section and findings: remove placeholders, SMOTE, baselines, and Mamba implementation claims.
{
  const s = slide(presentation, 19);
  setText(s, 4, "06");
  setText(s, 5, "Key Findings");
  setText(s, 6, "Summary of Deliverable 1 - Task 1 Presentation Takeaways");
}

{
  const s = slide(presentation, 20);
  setText(s, 7, "Key Findings");
  setText(s, 9, "What the data shows");
  setText(
    s,
    10,
    [
      "Demographics:",
      "  - Full Excel cohort: 166 subjects (93 PD, 73 HC).",
      "  - Fold files use a balanced 100-subject subset.",
      "  - Each fold has 60 train, 20 validation, 20 test files.",
      "",
      "Signal analysis:",
      "  - Raw VGRF traces show clear gait-cycle structure.",
      "  - Several PD examples show higher force variability.",
      "  - Variability plots are consistent with PD gait-rhythm disruption reported in the literature.",
      "",
      "Feature space:",
      "  - Features are z-score normalized before PCA/t-SNE.",
      "  - Left, right, and combined-foot views are analyzed separately.",
    ].join("\n"),
  );
  setText(s, 11, "Presentation takeaway");
  setText(
    s,
    12,
    [
      "Main message for Task 1:",
      "  - The dataset is usable and the official split structure is balanced.",
      "  - PCA/t-SNE show partial visual structure, not perfect PD/HC separation.",
      "  - Cohort effects remain visible and should be controlled in later modeling.",
      "",
      "Recommended next analysis checks:",
      "  - Regenerate final feature figures with validation included if the presentation must reflect all fold files.",
      "  - Keep train/validation/test boundaries fixed for future model evaluation.",
      "  - Report model results by fold and inspect cohort-specific behavior.",
    ].join("\n"),
  );
}

// Closing slide in English.
{
  const s = slide(presentation, 21);
  setText(s, 12, "Thank you\nfor your attention!");
}

removeEmptyStructuralPlaceholders(presentation);
updateVisibleFooters(presentation);

await fs.mkdir(outputDir, { recursive: true });
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPptx);

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
for (let index = 0; index < presentation.slides.count; index += 1) {
  const s = presentation.slides.getItem(index);
  await saveBlob(await presentation.export({ slide: s, format: "png", scale: 1 }), path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`));
  await fs.writeFile(path.join(layoutDir, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await (await presentation.export({ slide: s, format: "layout" })).text(), "utf8");
}

console.log(JSON.stringify({ finalPptx, slideCount: presentation.slides.count, previewDir, layoutDir }, null, 2));
