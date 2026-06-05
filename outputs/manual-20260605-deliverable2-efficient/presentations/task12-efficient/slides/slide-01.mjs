import { C, addFooter, metric } from "./common.mjs";
export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: "#F8FAFB" });
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 12, fill: C.teal });
  ctx.addText(slide, { x: 70, y: 78, w: 860, h: 58, text: "Parkinson's Disease Detection from Gait", fontSize: 38, bold: true, color: C.ink, typeface: ctx.fonts.title });
  ctx.addText(slide, { x: 72, y: 146, w: 850, h: 34, text: "Task 1 + Task 2 results with fold-safe Efficient TSFresh feature extraction", fontSize: 20, color: C.muted });
  ctx.addShape(slide, { x: 74, y: 220, w: 1030, h: 1, fill: C.line });
  metric(slide, ctx, 78, 268, 220, "83.0%", "Best mean test accuracy", C.teal);
  metric(slide, ctx, 328, 268, 220, "0.912", "Best mean test AUC", C.blue);
  metric(slide, ctx, 578, 268, 220, "5 folds", "Train / validation / test", C.green);
  metric(slide, ctx, 828, 268, 250, "84,430", "Efficient feature columns", C.gold);
  ctx.addText(slide, { x: 78, y: 430, w: 980, h: 68, text: "Main result: Linear SVM and Random Forest cross 80% under the corrected split protocol. TSFresh feature selection is useful in left/enhanced views, but the combined-view Linear SVM remains the strongest simple baseline.", fontSize: 21, color: C.ink });
  addFooter(slide, ctx, "Generated from results/tables/deliverable2_model_summary.csv");
  return slide;
}