import { C, addBase, addFooter, bar, table } from "./common.mjs";
export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Neural Grid", "Feature-vector neural models improve with tuning but trail classical models", "Grid: k_features in {100, 200, 400}, learning rate in {0.0003, 0.0008}, weight decay in {0.0001, 0.001}.");
  const bars = [
  {
    "label": "right PatchTST",
    "value": 0.79
  },
  {
    "label": "combined PatchTST",
    "value": 0.78
  },
  {
    "label": "enhanced PatchTST",
    "value": 0.77
  },
  {
    "label": "combined GRU",
    "value": 0.77
  },
  {
    "label": "right 1D ResNet",
    "value": 0.76
  },
  {
    "label": "left 1D ResNet",
    "value": 0.76
  }
];
  bars.forEach((row, i) => bar(slide, ctx, 84, 228 + i * 48, 300, row.label, row.value, 0.85, i === 0 ? C.blue : C.green));
  table(slide, ctx, [
  [
    "Scope",
    "Model",
    "Accuracy",
    "AUC"
  ],
  [
    "right",
    "PatchTST",
    "79.0%",
    "0.888"
  ],
  [
    "combined",
    "PatchTST",
    "78.0%",
    "0.866"
  ],
  [
    "enhanced",
    "PatchTST",
    "77.0%",
    "0.866"
  ],
  [
    "combined",
    "GRU",
    "77.0%",
    "0.826"
  ],
  [
    "right",
    "1D ResNet",
    "76.0%",
    "0.832"
  ]
], 730, 216, [130, 140, 105, 105], 42, { bodySize: 11.5, headerSize: 11, align: ["left", "left", "right", "right"] });
  ctx.addText(slide, { x: 84, y: 546, w: 520, h: 50, text: "PatchTST is the best neural option after grid tuning, but the feature-vector neural models do not surpass the best SVM/RF baselines.", fontSize: 17, color: C.ink });
  addFooter(slide, ctx, "Neural grid: results/tables/deliverable2_feature_neural_fold_metrics.csv");
  return slide;
}