import { C, addBase, addFooter } from "./common.mjs";
export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Evidence", "Summary charts and confusion matrices", "Rendered directly from the final merged fold-safe result tables.");
  await ctx.addImage(slide, { path: "/Users/keshavperiwal/Documents Local/Semester 2/Project Time Series/Data/results/figures/deliverable2_metric_summary.png", x: 68, y: 220, w: 560, h: 300, fit: "contain" });
  await ctx.addImage(slide, { path: "/Users/keshavperiwal/Documents Local/Semester 2/Project Time Series/Data/results/figures/deliverable2_best_confusion_matrices.png", x: 676, y: 220, w: 520, h: 300, fit: "contain" });
  ctx.addText(slide, { x: 82, y: 548, w: 1080, h: 46, text: "The confusion matrices are pooled across the test predictions for each best-by-scope model, after validation-selected model choices were fixed.", fontSize: 16, color: C.ink });
  addFooter(slide, ctx, "Figures: results/figures/deliverable2_metric_summary.png and deliverable2_best_confusion_matrices.png");
  return slide;
}