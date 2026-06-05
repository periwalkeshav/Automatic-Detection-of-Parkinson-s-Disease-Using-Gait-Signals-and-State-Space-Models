import { C, addBase, addFooter, table } from "./common.mjs";
export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Feature Selection", "TSFresh selection helps in specific views, not universally", "Comparison uses the same folds, same Efficient feature matrices, and validation-selected classifiers.");
  table(slide, ctx, [
  [
    "Scope",
    "Selector",
    "Model",
    "Acc.",
    "Sens.",
    "Spec.",
    "AUC"
  ],
  [
    "combined",
    "SelectKBest",
    "Linear SVM",
    "83.0%",
    "86.0%",
    "80.0%",
    "0.892"
  ],
  [
    "combined",
    "SelectKBest",
    "Random Forest",
    "82.0%",
    "84.0%",
    "80.0%",
    "0.892"
  ],
  [
    "enhanced",
    "select_features",
    "Random Forest + TSFresh",
    "83.0%",
    "82.0%",
    "84.0%",
    "0.901"
  ],
  [
    "enhanced",
    "SelectKBest",
    "Random Forest",
    "83.0%",
    "84.0%",
    "82.0%",
    "0.896"
  ],
  [
    "left",
    "select_features",
    "Linear SVM + TSFresh",
    "82.0%",
    "82.0%",
    "82.0%",
    "0.912"
  ],
  [
    "left",
    "select_features",
    "Random Forest + TSFresh",
    "82.0%",
    "80.0%",
    "84.0%",
    "0.910"
  ],
  [
    "right",
    "SelectKBest",
    "Random Forest",
    "83.0%",
    "86.0%",
    "80.0%",
    "0.902"
  ],
  [
    "right",
    "select_features",
    "Random Forest + TSFresh",
    "83.0%",
    "84.0%",
    "82.0%",
    "0.896"
  ]
], 64, 205, [130, 235, 230, 125, 125, 125, 105], 39, { bodySize: 11.5, headerSize: 11, align: ["left", "left", "left", "right", "right", "right", "right"] });
  ctx.addText(slide, { x: 78, y: 566, w: 940, h: 54, text: "Decision: keep both paths in the implementation. TSFresh select_features gives the best left/enhanced result; SelectKBest remains stronger for the combined Linear SVM.", fontSize: 17, color: C.ink });
  addFooter(slide, ctx, "Selector comparison: results/tables/deliverable2_feature_selection_comparison.csv");
  return slide;
}