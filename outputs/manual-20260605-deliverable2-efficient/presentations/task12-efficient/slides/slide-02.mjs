import { C, addBase, addFooter } from "./common.mjs";
export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addBase(slide, ctx, "Protocol", "Every fold follows train -> validation -> test", "No feature selection, scaling, tuning, or classifier fit uses validation/test labels.");
  const items = [
    ["Train", "Efficient TSFresh fixed calculators", "Fit imputer, z-score, feature selection, classifier"],
    ["Validation", "No fitting of learned transforms", "Select classifier hyperparameters and neural grid candidate"],
    ["Test", "Held out until final evaluation", "Report accuracy, sensitivity, specificity, AUC"],
  ];
  items.forEach((item, i) => {
    const x = 92 + i * 380;
    ctx.addShape(slide, { x, y: 250, w: 300, h: 170, fill: i === 0 ? "#EAF2F4" : i === 1 ? "#EEF3EA" : "#F6EFE7", line: ctx.line("#CAD6DC", 1) });
    ctx.addText(slide, { x: x + 22, y: 272, w: 250, h: 30, text: item[0], fontSize: 23, bold: true, color: i === 0 ? C.teal : i === 1 ? C.green : C.gold });
    ctx.addText(slide, { x: x + 22, y: 316, w: 250, h: 34, text: item[1], fontSize: 14, color: C.ink });
    ctx.addText(slide, { x: x + 22, y: 358, w: 250, h: 42, text: item[2], fontSize: 13, color: C.muted });
    if (i < 2) {
      ctx.addShape(slide, { x: x + 320, y: 316, w: 38, h: 4, fill: C.line });
      ctx.addShape(slide, { x: x + 352, y: 309, w: 16, h: 16, geometry: "triangle", fill: C.line });
    }
  });
  ctx.addText(slide, { x: 92, y: 474, w: 1000, h: 58, text: "Important distinction: EfficientFCParameters is a fixed feature calculator, not a learned estimator. The cached subject features avoid repeated computation; all learned preprocessing and selection are still fit inside the training split.", fontSize: 18, color: C.ink });
  addFooter(slide, ctx);
  return slide;
}