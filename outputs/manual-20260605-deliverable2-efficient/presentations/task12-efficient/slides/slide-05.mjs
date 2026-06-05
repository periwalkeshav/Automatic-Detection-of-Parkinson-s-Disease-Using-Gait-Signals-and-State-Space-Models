import { C, addBase, addFooter, table } from "./common.mjs";
export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Task 2", "Final five-fold test results", "Mean metrics across the five test folds; test data is not used for model selection.");
  table(slide, ctx, [
  [
    "Scope",
    "Best family",
    "Model",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "AUC"
  ],
  [
    "combined",
    "SelectKBest + classifier",
    "Linear SVM",
    "83.0%",
    "86.0%",
    "80.0%",
    "0.892"
  ],
  [
    "enhanced",
    "TSFresh selection + classifier",
    "Random Forest + TSFresh",
    "83.0%",
    "82.0%",
    "84.0%",
    "0.901"
  ],
  [
    "left",
    "TSFresh selection + classifier",
    "Linear SVM + TSFresh",
    "82.0%",
    "82.0%",
    "82.0%",
    "0.912"
  ],
  [
    "right",
    "SelectKBest + classifier",
    "Random Forest",
    "83.0%",
    "86.0%",
    "80.0%",
    "0.902"
  ]
], 66, 204, [120, 285, 235, 120, 120, 120, 100], 50, { bodySize: 12, headerSize: 11, align: ["left", "left", "left", "right", "right", "right", "right"] });
  ctx.addText(slide, { x: 80, y: 484, w: 455, h: 72, text: "Best accuracy: 83.0% across combined, enhanced, and right-foot views. Best AUC: 0.912 from left-foot Linear SVM + TSFresh selection.", fontSize: 18, bold: true, color: C.ink });
  ctx.addText(slide, { x: 620, y: 484, w: 490, h: 76, text: "Professor reference: Linear SVM around 80%. Corrected fold-safe Efficient features now exceed that while preserving validation/test separation.", fontSize: 18, color: C.ink });
  addFooter(slide, ctx, "Best models: results/tables/deliverable2_best_models_by_scope.csv");
  return slide;
}