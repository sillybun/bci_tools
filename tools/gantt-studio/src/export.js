import { DAY, WEEK, monday, categoryColor, imageState } from "./model.js";
const escape = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      })[c],
  );
const date = (t) => new Date(t).toISOString().slice(0, 10);
export function curve(x1, y1, x2, y2) {
  const bend = Math.max(32, Math.min(120, Math.abs(x2 - x1) * 0.5));
  // The final tangent tilts toward the target instead of flattening against its edge.
  const rise = Math.sign(y2-y1) * Math.min(bend * 0.55, Math.abs(y2-y1) * 0.4);
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2-rise}, ${x2} ${y2}`;
}
function short(s, limit) {
  return [...String(s ?? "")].length > limit
    ? [...String(s)].slice(0, Math.max(1, limit - 1)).join("") + "…"
    : String(s ?? "");
}
export function projectSVG(state) {
  state = imageState(state);
  const palette = state.categories.length ? state.categories.map(categoryColor) : ["#6961d9"];
  const times = [
    ...state.tasks.flatMap((t) => [t.start, t.end]),
    ...(state.milestones ?? []).map((m) => m.time),
  ];
  if (!times.length) times.push(Date.now());
  const origin = monday(Math.min(...times)) - WEEK;
  const latest = Math.max(...times);
  // Round up to the enclosing week boundary, then leave exactly one spare column.
  const end = monday(latest) + (latest === monday(latest) ? 1 : 2) * WEEK;
  const count = Math.ceil((end - origin) / WEEK);
  const weekWidth = Math.max(14, Math.min(150, 1800 / count));
  const compact=!!state.settings?.compact, rh=compact?32:64, pad=compact?6:12, th=compact?24:48;
  const hh=state.settings?.headerHeight??60;
  const left = 40+(state.settings?.columnWidth??224),
    width = left + count * weekWidth + 40,
    top = 160+hh+((state.milestones?.length??0)>1?48:(state.milestones?.length??0)===1?24:0);
  const positions = new Map(),
    sections = [];
  let y = top;
  for (const c of state.categories) {
    sections.push({ c, y });
    for (const t of state.tasks.filter((t) => t.categoryId === c.id))
      positions.set(t.id, {
        x: left + ((t.start - origin) / WEEK) * weekWidth,
        y: y + pad + t.row * rh,
        w: ((t.end - t.start) / WEEK) * weekWidth,
      });
    y += c.rows * rh + pad*2;
  }
  const height = Math.max(420, y + 94),
    x = (t) => left + ((t - origin) / WEEK) * weekWidth;
  const fontScale = (state.settings?.fontSize ?? 14) / 14;
  const text = (tx, ty, s, size = 13, fill = "#626d83", extra = "") =>
    `<text x="${tx}" y="${ty}" font-size="${size * fontScale}" fill="${fill}" ${extra}>${escape(s)}</text>`;
  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.ceil(width)}" height="${height}" viewBox="0 0 ${Math.ceil(width)} ${height}"><defs>${palette.map((color, i) => `<marker id="arrow${i}" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M2 1.5 L8 5 L2 8.5" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>`).join("")}</defs><rect width="100%" height="100%" fill="#f5f7fb"/><g font-family="Arial, Noto Sans SC, Microsoft YaHei, sans-serif"><rect x="20" y="20" width="${width - 40}" height="${height - 40}" rx="18" fill="white"/>`,
    text(
      48,
      60,
      "GANTT STUDIO  /  PROJECT TIMELINE",
      11,
      "#8f86b4",
      'letter-spacing="2"',
    ),
    text(
      48,
      105,
      short(state.name || "项目时间线", 32),
      30,
      "#293246",
      'font-weight="700"',
    ),
    text(
      48,
      137,
      `${date(Math.min(...times))} — ${date(Math.max(...times))}   ·   ${state.tasks.length} 个任务   ·   ${state.milestones?.length ?? 0} 个关键点`,
      13,
    ),
    `<rect x="40" y="160" width="${width - 80}" height="${top-160}" rx="8" fill="#f4f3fc"/>`,
    text(56, 160+hh/2+5, "任务类别", 13, "#605c82"),
  ];
  const stride = Math.max(1, Math.ceil(90 / weekWidth));
  for (let i = 0; i <= count; i++) {
    const px = left + i * weekWidth;
    svg.push(`<path d="M${px} 160 V${y}" stroke="#eceef5"/>`);
    if (i < count && i % stride === 0)
      svg.push(
        text(px + 8, 160+hh/2-8, date(origin + i * WEEK).slice(0, 7), 10, "#9b9fb1"),
        text(px + 8, 160+hh/2+12, date(origin + i * WEEK).slice(5), 12, "#69628b"),
      );
  }
  sections.forEach(({ c, y: cy }, index) => {
    svg.push(
      `<path d="M40 ${cy} H${width - 40}" stroke="#e7eaf2"/>`,
      text(
        56,
        cy + (compact?17:30),
        short(c.name, Math.floor((left-72)/18)),
        14,
        palette[index % palette.length],
        'font-weight="600"',
      ),
      text(56, cy + (compact?34:53), short(c.description, Math.floor((left-72)/14)), compact?9:11, "#939bad"),
    );
  });
  (state.milestones ?? [])
    .slice()
    .sort((a, b) => a.time - b.time)
    .forEach((m, i) => {
      const px = x(m.time),
        ly = 160+hh+11 + (i % 2) * 24;
      svg.push(
        `<path d="M${px} ${top} V${y}" stroke="#cf9250" stroke-dasharray="3 5" opacity=".65"/><path d="M${px} ${ly - 5} l5 5 -5 5 -5 -5Z" fill="#cf9250"/>`,
        text(
          Math.min(width - 180, Math.max(left, px + 9)),
          ly + 3,
          short(m.title, 16),
          14,
          "#b57c3b",
          'font-weight="600"',
        ),
      );
    });
  for (const e of state.dependencies ?? []) {
    const a = positions.get(e.from),
      b = positions.get(e.to);
    const index =
      Math.max(
        0,
        state.categories.findIndex(
          (c) => c.id === state.tasks.find((t) => t.id === e.from)?.categoryId,
        ),
      ) % palette.length;
    if (a && b)
      svg.push(
        `<path d="${curve(a.x + a.w + 1, a.y + th/2, b.x - 3, b.y + th/2)}" fill="none" stroke="${palette[index]}" stroke-width="1.6" stroke-linecap="round" stroke-dasharray="5 5" marker-end="url(#arrow${index})"/>`,
      );
  }
  state.tasks.forEach((t, index) => {
    const p = positions.get(t.id),
      color =
        palette[
          state.categories.findIndex((c) => c.id === t.categoryId) %
            palette.length
        ],
      w = Math.max(3, p.w),
      titleOnly = state.settings?.showTaskDates === false,
      hideOverflow = !!state.settings?.hideOverflowText;
    // Explicit alphabetic baseline keeps titles centered in viewers that ignore dominant-baseline.
    svg.push(
      `<defs><clipPath id="task-${index}"><rect x="${p.x + 4}" y="${p.y}" width="${Math.max(0, w - 8)}" height="${th}"/></clipPath></defs><rect x="${p.x}" y="${p.y}" width="${w}" height="${th}" rx="8" fill="${color}" fill-opacity=".13" stroke="${color}" stroke-opacity=".35"/><g${hideOverflow ? ` clip-path="url(#task-${index})"` : ' data-task-text="true"'}>`,
      text(
        titleOnly && hideOverflow ? p.x + w / 2 : p.x + 12,
        titleOnly ? p.y + th/2 + 15 * fontScale * 0.35 : p.y + (compact?10:21),
        hideOverflow ? short(
          t.title,
          Math.floor((w - 20) / ((titleOnly ? 15 : 13) * fontScale)),
        ) : t.title,
        titleOnly ? 15 : (compact?9:12),
        color,
        titleOnly
          ? `font-weight="600" text-anchor="${hideOverflow ? "middle" : "start"}"`
          : 'font-weight="600"',
      ),
      state.settings?.showTaskDates === false
        ? ""
        : text(
            p.x + 12,
            p.y + (compact?21:39),
            `${date(t.start).slice(5)} — ${date(t.end).slice(5)}`,
            compact?8:10,
            color,
          ),
      "</g>",
    );
  });
  svg.push(
    `<path d="M54 ${height-58} l5 5 -5 5 -5 -5Z" fill="#cf9250"/>`,
    text(66,height-48,"关键点",11,"#8b93a6"),
    `<path d="M145 ${height-53} H190" fill="none" stroke="${palette[0]}" stroke-width="1.6" stroke-linecap="round" stroke-dasharray="5 5" marker-end="url(#arrow0)"/>`,
    text(202,height-48,"完成后开始依赖",11,"#8b93a6"),
    text(width - 235, height - 48, "Gantt Studio · 计划有形", 11, "#9b91ba"),
    "</g></svg>",
  );
  return { svg: svg.join(""), width: Math.ceil(width), height };
}
export function download(blob, name) {
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
export async function exportImage(state, format) {
  const { svg, width, height } = projectSVG(state),
    blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const safeName = (state.name || "gantt-timeline")
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, "_")
    .slice(0, 100);
  const name = `${safeName}-${date(Date.now())}`;
  if (format === "svg") {
    download(blob, name + ".svg");
    return;
  }
  // Bound canvas memory while retaining 2x resolution for ordinary projects.
  const ratio = Math.min(
    2,
    8192 / width,
    16384 / height,
    Math.sqrt(32_000_000 / (width * height)),
  );
  if (ratio < 0.25) throw Error("项目过大，建议选择 SVG 矢量图导出");
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    await document.fonts.ready;
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(width * ratio);
    canvas.height = Math.ceil(height * ratio);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw Error("无法创建图片画布");
    ctx.scale(ratio, ratio);
    ctx.drawImage(img, 0, 0, width, height);
    const output = await new Promise((resolve) =>
      canvas.toBlob(
        resolve,
        format === "jpg" ? "image/jpeg" : "image/png",
        0.95,
      ),
    );
    if (!output) throw Error("图片生成失败，请改用 SVG");
    download(output, name + (format === "jpg" ? ".jpg" : ".png"));
  } finally {
    URL.revokeObjectURL(url);
  }
}
