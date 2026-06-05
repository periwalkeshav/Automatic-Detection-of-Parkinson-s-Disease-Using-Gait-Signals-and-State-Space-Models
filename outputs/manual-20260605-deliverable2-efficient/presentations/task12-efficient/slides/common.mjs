
export const data = {
  "cover": {
    "bestAcc": "83.0%",
    "bestAuc": "0.912"
  },
  "bestRows": [
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
  ],
  "auditRows": [
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
  ],
  "selectorRows": [
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
  ],
  "neuralRows": [
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
  ],
  "neuralBars": [
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
  ]
};

export const C = {
  ink: "#17202A",
  muted: "#5C6670",
  soft: "#EEF2F5",
  line: "#C9D2DA",
  teal: "#2F5F73",
  blue: "#3D6EA8",
  green: "#4F7D61",
  gold: "#B9822E",
  red: "#A95B50",
  white: "#FFFFFF",
};

export function addBase(slide, ctx, kicker, title, subtitle = "") {
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 720, fill: C.white });
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 10, fill: C.teal });
  ctx.addText(slide, { x: 56, y: 38, w: 230, h: 22, text: kicker.toUpperCase(), fontSize: 13, bold: true, color: C.teal, typeface: ctx.fonts.body });
  ctx.addText(slide, { x: 56, y: 66, w: 860, h: 78, text: title, fontSize: 30, bold: true, color: C.ink, typeface: ctx.fonts.title });
  if (subtitle) ctx.addText(slide, { x: 56, y: 150, w: 900, h: 42, text: subtitle, fontSize: 15, color: C.muted });
  ctx.addText(slide, { x: 1090, y: 674, w: 130, h: 18, text: "Task 1 + Task 2", fontSize: 12, color: C.muted, align: "right" });
}

export function addFooter(slide, ctx, text = "Fold-safe Efficient TSFresh pipeline") {
  ctx.addText(slide, { x: 56, y: 674, w: 780, h: 18, text, fontSize: 11, color: C.muted });
}

export function metric(slide, ctx, x, y, w, value, label, color = C.teal) {
  ctx.addShape(slide, { x, y, w, h: 96, fill: C.soft, line: ctx.line("#D8E0E6", 1) });
  ctx.addText(slide, { x: x + 18, y: y + 14, w: w - 36, h: 36, text: value, fontSize: 30, bold: true, color });
  ctx.addText(slide, { x: x + 18, y: y + 56, w: w - 36, h: 28, text: label, fontSize: 13, color: C.muted });
}

export function table(slide, ctx, rows, x, y, widths, rowH, opts = {}) {
  const headerFill = opts.headerFill ?? C.teal;
  const head = rows[0];
  let cx = x;
  head.forEach((cell, i) => {
    ctx.addShape(slide, { x: cx, y, w: widths[i], h: rowH, fill: headerFill });
    ctx.addText(slide, { x: cx + 8, y: y + 9, w: widths[i] - 16, h: rowH - 12, text: cell, fontSize: opts.headerSize ?? 12, bold: true, color: C.white, align: opts.align?.[i] ?? "left" });
    cx += widths[i];
  });
  rows.slice(1).forEach((row, r) => {
    cx = x;
    const fill = r % 2 === 0 ? "#F7F9FA" : C.white;
    row.forEach((cell, i) => {
      ctx.addShape(slide, { x: cx, y: y + rowH * (r + 1), w: widths[i], h: rowH, fill, line: ctx.line("#DDE5EA", 1) });
      ctx.addText(slide, { x: cx + 8, y: y + rowH * (r + 1) + 8, w: widths[i] - 16, h: rowH - 11, text: String(cell), fontSize: opts.bodySize ?? 12, color: C.ink, bold: opts.boldRows?.includes(r) ?? false, align: opts.align?.[i] ?? "left" });
      cx += widths[i];
    });
  });
}

export function bar(slide, ctx, x, y, w, label, value, max = 1, color = C.blue) {
  ctx.addText(slide, { x, y: y - 2, w: 165, h: 24, text: label, fontSize: 13, color: C.ink });
  ctx.addShape(slide, { x: x + 175, y, w, h: 18, fill: "#E6EBEF" });
  ctx.addShape(slide, { x: x + 175, y, w: Math.max(2, w * value / max), h: 18, fill: color });
  ctx.addText(slide, { x: x + 185 + w, y: y - 2, w: 70, h: 24, text: (value * 100).toFixed(1) + "%", fontSize: 13, bold: true, color });
}
