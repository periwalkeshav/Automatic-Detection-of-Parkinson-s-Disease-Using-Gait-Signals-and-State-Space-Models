import { C, addBase, addFooter } from "./common.mjs";
export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Reproducibility", "Files generated for review and rerun", "The notebook and runner are set up so the experiment can be rerun without changing the protocol.");
  const lines = [
    ["Notebook", "Task1_Task2_Efficient_FoldSafe.ipynb"],
    ["Runner", "scripts/run_deliverable2.py -> scripts/run_deliverable2_foldsafe.py"],
    ["Feature matrices", "results/fold_features/Fold_*/{training,validation,test}_feature_matrix_efficient.csv"],
    ["Result tables", "results/tables/deliverable2_model_summary.csv"],
    ["Feature selection comparison", "results/tables/deliverable2_feature_selection_comparison.csv"],
  ];
  lines.forEach((row, i) => {
    const y = 214 + i * 62;
    ctx.addText(slide, { x: 96, y, w: 250, h: 28, text: row[0], fontSize: 15, bold: true, color: C.teal });
    ctx.addText(slide, { x: 360, y, w: 760, h: 28, text: row[1], fontSize: 15, color: C.ink });
    ctx.addShape(slide, { x: 96, y: y + 40, w: 1010, h: 1, fill: "#E3E9EE" });
  });
  ctx.addText(slide, { x: 96, y: 562, w: 970, h: 44, text: "Recommended presentation claim: the corrected fold-safe protocol is now defensible, and Efficient TSFresh + classical classifiers is the current best non-Mamba baseline.", fontSize: 17, bold: true, color: C.ink });
  addFooter(slide, ctx);
  return slide;
}