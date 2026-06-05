import { C, addBase, addFooter, table } from "./common.mjs";
export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Task 1", "Fold distribution and feature extraction audit", "Efficient TSFresh produced consistent fold-local matrices for all train/validation/test splits.");
  table(slide, ctx, [
  [
    "Fold",
    "Split",
    "Rows",
    "Feature cols",
    "Protocol note"
  ],
  [
    "Fold_1",
    "training",
    "60",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ],
  [
    "Fold_1",
    "validation",
    "20",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ],
  [
    "Fold_1",
    "test",
    "20",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ],
  [
    "Fold_2",
    "training",
    "60",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ],
  [
    "Fold_2",
    "validation",
    "20",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ],
  [
    "Fold_2",
    "test",
    "20",
    "84,430",
    "fixed calculators; train-fitted learned steps"
  ]
], 74, 224, [150, 150, 120, 190, 420], 46, { bodySize: 13, headerSize: 12, align: ["left", "left", "right", "right", "left"] });
  ctx.addShape(slide, { x: 78, y: 546, w: 1050, h: 62, fill: "#F7F9FA", line: ctx.line("#D9E2E8", 1) });
  ctx.addText(slide, { x: 98, y: 562, w: 1010, h: 34, text: "Audit check: every fold has 60 training, 20 validation, and 20 test subjects. Each Efficient matrix has 84,430 feature columns before train-fitted filtering/selection.", fontSize: 16, color: C.ink });
  addFooter(slide, ctx, "Feature audit: results/tables/deliverable2_fold_safe_feature_audit.csv");
  return slide;
}