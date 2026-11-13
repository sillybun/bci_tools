import "./style.css";
import { ProjectHistory } from "./history.js";
import {
  loadLibrary,
  persistProject,
  projectState,
  LIBRARY_KEY,
} from "./projects.js";
import {
  categoryColor,
  imageState,
  DAY,
  WEEK,
  monday,
  snap,
  placeTask,
  validate,
  linkTasks,
  scheduleChange,
  enforceDependencies,
  dependencyConflicts,
  settleRows,
  pruneDependencies,
  deleteEmptyRow,
  assertTimeRange,
} from "./model.js";
import { curve, exportImage, download } from "./export.js";
const $ = (s) => document.querySelector(s),
  uid = () => crypto.randomUUID();
const today = Date.UTC(
  new Date().getFullYear(),
  new Date().getMonth(),
  new Date().getDate(),
);
const base = monday(today);
const DEFAULT_CATEGORY_COLORS = ["#6961d9", "#299c91", "#cf9250", "#568acb", "#bf6f9c", "#e06b5f", "#7c8f3f", "#354c7c"];
const initial = {
  categories: [
    {
      id: "design",
      name: "产品与设计",
      description: "从想法到清晰的解决方案",
      rows: 2,
    },
    {
      id: "build",
      name: "研发与实现",
      description: "让每一个细节落地",
      rows: 2,
    },
    {
      id: "launch",
      name: "测试与发布",
      description: "准备好，迎接下一步",
      rows: 1,
    },
  ],
  tasks: [
    {
      id: "a",
      categoryId: "design",
      row: 0,
      title: "需求调研与分析",
      notes: "收集需求，明确项目的目标与范围。",
      start: base - WEEK,
      end: base + WEEK,
    },
    {
      id: "b",
      categoryId: "design",
      row: 1,
      title: "界面设计",
      notes: "",
      start: base,
      end: base + 2 * WEEK,
    },
    {
      id: "c",
      categoryId: "build",
      row: 0,
      title: "核心功能开发",
      notes: "",
      start: base + WEEK,
      end: base + 4 * WEEK,
    },
    {
      id: "d",
      categoryId: "build",
      row: 1,
      title: "跨平台适配",
      notes: "",
      start: base + 2 * WEEK,
      end: base + 5 * WEEK,
    },
    {
      id: "e",
      categoryId: "launch",
      row: 0,
      title: "测试与发布",
      notes: "",
      start: base + 5 * WEEK,
      end: base + 7 * WEEK,
    },
  ],
};
let state = initial,
  library,
  libraryBlocked = false,
  saveError = false;
try {
  library = loadLibrary(localStorage, initial);
  state = projectState(library);
} catch {
  libraryBlocked = true;
  saveError = true;
  state = structuredClone(initial);
}
state.name ??= "项目时间线";
state.settings ??= { showTaskDates: true, fontSize: 14 };
state.dependencies ??= [];
state.milestones ??= [];
let scale = 112 / WEEK,
  origin = base - 12 * WEEK,
  weeks = 104,
  drag = null;
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const date = (t) => new Date(t).toISOString().slice(0, 10),
  stamp = (t) => new Date(t).toISOString().slice(0, 16).replace("T", " ");
const history = new ProjectHistory();
function save(recordHistory = true) {
  try {
    if (libraryBlocked) throw Error("项目库无法读取");
    library = persistProject(localStorage, library, state);
    if (recordHistory) history.record(library.activeId, state);
    updateHistoryButtons();
    $("#saved").textContent = "● 所有更改已保存";
  } catch {
    $("#saved").textContent = "保存失败，请导出备份";
    return false;
  }
  renderStats();
  renderProjectList();
  return true;
}
$("#app").innerHTML =
  `<aside class="rail"><div class="brand">g<span>·</span></div><button class="rail-item active" id="projectsButton" title="项目列表" aria-label="项目列表" aria-expanded="false">▤</button><div class="rail-bottom">GS</div></aside><main><header><div class="crumb">工作空间 <span>/</span> 项目计划</div><div id="saved">● 本地自动保存</div></header><section class="intro"><div><div class="eyebrow">YOUR PLANS, IN PERSPECTIVE</div><h1><span id="projectTitle" tabindex="0" title="双击修改项目名称"></span><input id="projectNameInput" aria-label="项目名称" maxlength="100" hidden><span class="badge">甘特图</span></h1><p>把想法排进日程，让每一步都清晰可见。</p></div><div class="actions"><button id="newProject">＋ 新建项目</button><button id="import">↥ 导入</button><div class="export-control"><button id="export" aria-haspopup="menu" aria-expanded="false">↧ 导出项目 ▾</button><div id="exportMenu" class="export-menu" role="menu" hidden><button role="menuitem" data-export="json">JSON <small>可编辑的项目备份</small></button><button role="menuitem" data-export="png">PNG <small>高清图片</small></button><button role="menuitem" data-export="jpg">JPG <small>高清照片格式</small></button><button role="menuitem" data-export="svg">SVG <small>可缩放矢量图</small></button></div></div><button class="primary" id="addCategory">＋ 新建类别</button></div></section><button id="toggleStats" class="stats-toggle" aria-expanded="true" aria-controls="projectStats">▾ 项目概览</button><section class="stats" id="projectStats"><div><span class="stat-icon purple">▤</span><span><b id="taskCount"></b><small>计划任务</small></span></div><div><span class="stat-icon green">▦</span><span><b id="categoryCount"></b><small>任务类别</small></span></div><div><span class="stat-icon orange">◷</span><span><b id="duration"></b><small>计划跨度</small></span></div><p>好的计划，始于清晰的时间线。<span>PLAN WITH CLARITY</span></p></section><section class="board"><div class="display-settings"><label><input type="checkbox" id="compactMode"> 紧凑模式</label><label><input type="checkbox" id="showTaskDates" checked> 显示任务时间区间</label><label><input type="checkbox" id="hideOverflowText"> 隐藏越界文本</label><label><input type="checkbox" id="lockDependencies"> 锁定依赖关系</label><label>文字大小 <select id="fontSize"><option value="12">小</option><option value="14" selected>标准</option><option value="16">大</option><option value="18">特大</option></select></label><span>同时应用于 PNG / JPG / SVG 导出</span></div><div class="toolbar"><div><span class="live-dot"></span><strong>时间规划</strong><span class="muted">按周视图</span></div><div class="tools"><button id="undo" title="撤销 · Ctrl / ⌘ Z" aria-label="撤销" disabled>↶</button><button id="redo" title="重做 · Ctrl / ⌘ Shift Z" aria-label="重做" disabled>↷</button><button id="addMilestone">◇ 关键点</button><button id="today">今天</button><span class="divider"></span><button id="zoomOut" aria-label="缩小">−</button><span id="zoomLabel">100%</span><button id="zoomIn" aria-label="放大">＋</button><span class="week-label">周 ▾</span></div></div><div class="viewport" id="viewport"><div id="canvas"></div></div><div class="board-footer"><span>＋ 双击空白处新建任务</span><span>拖动空白处平移 · Ctrl / ⌘ + 滚轮缩放</span></div></section><footer><span><i></i> 默认按周吸附 <kbd>Ctrl</kbd> / <kbd>⌘</kbd> 按住自由调整</span><span>Shift 拖动任务建立依赖 · 双击编辑</span></footer></main><div id="projectBackdrop" hidden></div><aside id="projectPanel" hidden aria-label="本地项目"><div class="project-panel-header"><h2>我的项目</h2><button id="closeProjects" aria-label="关闭项目列表">×</button></div><p>保存在此设备 · 自动保存</p><button id="panelNewProject" class="primary">＋ 新建项目</button><div id="projectList"></div></aside><dialog id="editor"></dialog><div id="notice" role="status"></div><input type="file" id="file" accept="application/json,.json" hidden>`;
function renderStats() {
  $("#taskCount").textContent = state.tasks.length;
  $("#categoryCount").textContent = state.categories.length;
  $("#duration").textContent = state.tasks.length
    ? Math.ceil(
        (Math.max(...state.tasks.map((t) => t.end)) -
          Math.min(...state.tasks.map((t) => t.start))) /
          DAY,
      ) + " 天"
    : "—";
}
function columnWidth(){return state.settings?.columnWidth??224;}
function headerHeight(){return state.settings?.headerHeight??60;}
function rowHeight(){return state.settings?.compact?32:64;}
function rowPadding(){return state.settings?.compact?6:12;}
function render() {
  applyDisplaySettings();
  $("#projectTitle").textContent = state.name;
  const width = weeks * WEEK * scale;
  let html = `<div class="time-header"><div class="corner">任务类别 <span>${state.categories.length} 个类别</span></div><div class="ticks" style="width:${width}px">`;
  for (let i = 0; i < weeks; i++) {
    const t = origin + i * WEEK;
    html += `<div class="tick ${t <= today && today < t + WEEK ? "current" : ""}" style="width:${WEEK * scale}px"><small>${new Date(t).getUTCFullYear()} / ${String(new Date(t).getUTCMonth() + 1).padStart(2, "0")}</small><b>${date(t).slice(5).replace("-", " / ")}</b><span>周一</span></div>`;
  }
  html += `</div></div><div class="milestone-row"><div class="category-label">◇ 重要时间节点</div><div class="milestone-track" style="width:${width}px">`;
  state.milestones.forEach((m) => {
    html += `<button class="milestone" data-milestone="${esc(m.id)}" style="left:${(m.time - origin) * scale}px" title="${esc(m.title)} · ${stamp(m.time)}">◆ ${esc(m.title)} <small>${date(m.time).slice(5)}</small></button>`;
  });
  html += "</div></div>";
  state.categories.forEach((c, index) => {
    const color = categoryColor(c, index);
    const swatches = DEFAULT_CATEGORY_COLORS.map(v => `<button type="button" class="color-swatch${v === color ? " selected" : ""}" data-palette-color="${esc(c.id)}" data-value="${v}" style="--swatch:${v}" aria-label="选择 ${v}"></button>`).join("");
    html += `<section class="category${c.exportVisible === false ? " export-hidden" : ""}" data-category="${esc(c.id)}" style="height:${c.rows * rowHeight() + rowPadding()*2}px;--accent:${color}"><div class="category-label"><div><button class="category-grip" data-reorder="${esc(c.id)}" aria-label="${esc(c.name)}：拖动排序" title="拖动排序（方向键上下移动）">⠿</button><button type="button" class="category-color" data-color="${esc(c.id)}" aria-label="${esc(c.name)}：类别颜色" aria-expanded="false" title="选择类别颜色" style="--accent:${color}"></button><div class="color-palette" data-palette="${esc(c.id)}" popover><span>默认颜色</span>${swatches}<label>自定义<input type="color" data-custom-color="${esc(c.id)}" value="${color}" aria-label="自定义类别颜色"></label></div><strong>${esc(c.name)}</strong><button class="category-visibility" data-visibility="${esc(c.id)}" aria-label="${esc(c.name)}：图片导出可见" aria-pressed="${c.exportVisible !== false}" title="${c.exportVisible === false ? "在图片中显示此类别" : "从图片中隐藏此类别"}"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>${c.exportVisible === false ? '<path d="m3 3 18 18"/>' : ""}</svg></button><button class="edit-category" data-id="${esc(c.id)}" aria-label="编辑类别"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" focusable="false"><circle cx="4" cy="7" r="1" fill="currentColor"/><circle cx="7" cy="7" r="1" fill="currentColor"/><circle cx="10" cy="7" r="1" fill="currentColor"/></svg></button></div><p>${esc(c.description)}</p><small>${state.tasks.filter((t) => t.categoryId === c.id).length} 个任务 · ${c.rows} 行</small></div><div class="track" data-category="${esc(c.id)}" style="width:${width}px;background-size:${WEEK * scale}px ${rowHeight()}px"><div class="edge top">释放以在顶部新增一行</div><div class="edge bottom">释放以在底部新增一行</div>`;
    for (let row = 0; row < c.rows; row++) {
      if (
        c.rows > 1 &&
        !state.tasks.some((t) => t.categoryId === c.id && t.row === row)
      )
        html += `<div class="row-action" style="top:${rowPadding() + row * rowHeight()}px"><button data-delete-row="${row}" data-row-category="${esc(c.id)}" title="删除此空行" aria-label="${esc(c.name)}：删除第 ${row + 1} 空行">− 删除空行 <small>第 ${row + 1} 行</small></button></div>`;
    }
    state.tasks
      .filter((t) => t.categoryId === c.id)
      .forEach((t) => {
        html += `<div class="task" data-task="${esc(t.id)}" style="left:${(t.start - origin) * scale}px;top:${rowPadding() + t.row * rowHeight()}px;width:${Math.max(3, (t.end - t.start) * scale)}px" title="${esc(t.title)}\n${stamp(t.start)} → ${stamp(t.end)}"><span class="handle left" data-resize="start"></span><strong>${esc(t.title)}</strong>${state.settings.showTaskDates ? `<small>${date(t.start).slice(5)} — ${date(t.end).slice(5)}</small>` : ""}<span class="handle right" data-resize="end"></span></div>`;
      });
    html += `<div class="today-line" style="left:${(today - origin) * scale}px"></div></div></section>`;
  });
  html += `<div class="empty-row"><div class="category-label"><button id="inlineCategory">＋ 添加类别</button></div><div style="width:${width}px"></div></div>`;
  $("#canvas").innerHTML = html;
  document.querySelectorAll('.corner,.category-label').forEach(el=>el.insertAdjacentHTML('beforeend','<span class="column-resizer" data-layout="columnWidth" role="separator" tabindex="0" aria-label="调整类别列宽" aria-orientation="vertical"></span>'));
  $('.time-header').insertAdjacentHTML('beforeend','<span class="header-resizer" data-layout="headerHeight" role="separator" tabindex="0" aria-label="调整时间表头高度" aria-orientation="horizontal"></span>');
  $("#zoomLabel").textContent = Math.round(((scale * WEEK) / 112) * 100) + "%";
  renderStats();
  drawOverlay();
}
function timeAt(e) {
  const rect = $("#viewport").getBoundingClientRect();
  return (
    origin + (e.clientX - rect.left + $("#viewport").scrollLeft - columnWidth()) / scale
  );
}
function ensureRange() {
  const times = [
    ...state.tasks.flatMap((t) => [t.start, t.end]),
    ...state.milestones.map((m) => m.time),
  ];
  const min = Math.min(base - 12 * WEEK, ...times) - 2 * WEEK,
    max = Math.max(base + 92 * WEEK, ...times) + 4 * WEEK;
  const next = monday(min);
  const shift = (origin - next) * scale;
  origin = next;
  weeks = Math.ceil((max - origin) / WEEK);
  render();
  $("#viewport").scrollLeft += shift;
}
function goToday() {
  $("#viewport").scrollLeft = Math.max(0, (today - origin - 2 * WEEK) * scale);
}
function zoom(factor, x = ($("#viewport").clientWidth - columnWidth()) / 2) {
  const v = $("#viewport"),
    time = origin + (v.scrollLeft + x) / scale;
  scale = Math.max(35 / WEEK, Math.min(420 / WEEK, scale * factor));
  render();
  v.scrollLeft = (time - origin) * scale - x;
}
$("#zoomIn").onclick = () => zoom(1.25);
$("#zoomOut").onclick = () => zoom(0.8);
$("#today").onclick = goToday;
$("#viewport").addEventListener(
  "wheel",
  (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      zoom(
        e.deltaY < 0 ? 1.1 : 1 / 1.1,
        e.clientX - $("#viewport").getBoundingClientRect().left - columnWidth(),
      );
    }
  },
  { passive: false },
);
function showEditor(html, submit) {
  const d = $("#editor");
  d.innerHTML = `<form>${html}<div class="dialog-actions"><button type="button" id="cancel">取消</button><button class="primary" type="submit">保存更改</button></div></form>`;
  $("#cancel").onclick = () => d.close();
  d.querySelector("form").onsubmit = (e) => {
    e.preventDefault();
    if (submit(new FormData(e.target)) !== false) d.close();
  };
  d.showModal();
}
let noticeTimer;
function notify(message) {
  $("#notice").textContent = message;
  $("#notice").classList.add("show");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => $("#notice").classList.remove("show"), 4500);
}
function drawOverlay(pointer) {
  $("#connections")?.remove();
  const canvas = $("#canvas"),
    rect = canvas.getBoundingClientRect(),
    width = canvas.scrollWidth,
    height = canvas.scrollHeight;
  const points = new Map(
    [...canvas.querySelectorAll(".task")].map((el) => {
      const r = el.getBoundingClientRect();
      return [
        el.dataset.task,
        {
          left: r.left - rect.left,
          right: r.right - rect.left,
          y: r.top - rect.top + r.height / 2,
        },
      ];
    }),
  );
  const colors = state.categories.map(categoryColor);
  let paths = `<defs>${colors.map((color, i) => `<marker id="depArrow${i}" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M2 1.5 L8 5 L2 8.5" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>`).join("")}</defs>`;
  const sourceIndex = (id) =>
    Math.max(
      0,
      state.categories.findIndex(
        (c) => c.id === state.tasks.find((t) => t.id === id)?.categoryId,
      ),
    ) % colors.length;
  const conflicts = new Set(dependencyConflicts(state));
  const conflictedTasks = new Set([...conflicts].flatMap(e => [e.from, e.to]));
  canvas.querySelectorAll(".task").forEach(el => {
    const conflict = conflictedTasks.has(el.dataset.task);
    el.classList.toggle("dependency-conflict", conflict);
    el.setAttribute("aria-invalid", String(conflict));
  });
  for (const edge of state.dependencies) {
    const a = points.get(edge.from),
      b = points.get(edge.to);
    if (a && b) {
      const index = sourceIndex(edge.from);
      paths += `<path class="dependency-line${conflicts.has(edge) ? " dependency-conflict" : ""}" style="stroke:${colors[index]}" d="${curve(a.right + 1, a.y, b.left - 3, b.y)}" marker-end="url(#depArrow${index})"/>`;
    }
  }
  for (const m of state.milestones) {
    const x = columnWidth() + (m.time - origin) * scale;
    paths += `<path class="milestone-line" d="M ${x} ${headerHeight()+48} V ${height - 74}"/>`;
  }
  if (pointer && drag?.type === "link") {
    const a = points.get(drag.task.id);
    if (a)
      paths += `<path class="dependency-line preview" style="stroke:${colors[sourceIndex(drag.task.id)]}" d="${curve(a.right, a.y, pointer.clientX - rect.left, pointer.clientY - rect.top)}" marker-end="url(#depArrow${sourceIndex(drag.task.id)})"/>`;
  }
  canvas.insertAdjacentHTML(
    "beforeend",
    `<svg id="connections" width="${width}" height="${height}" aria-hidden="true">${paths}</svg>`,
  );
}
function milestoneEditor(
  m = { id: uid(), title: "", time: today },
  isNew = true,
) {
  showEditor(
    `<h2>${isNew ? "添加关键点" : "编辑关键点"}</h2><label>关键点名称<input name="title" maxlength="120" value="${esc(m.title)}" required autofocus></label><label>重要时间<input type="datetime-local" name="time" value="${new Date(m.time).toISOString().slice(0, 16)}" required></label><p id="formError" class="form-error"></p>${isNew ? "" : '<button type="button" id="deleteMilestone" class="danger">删除关键点</button>'}`,
    (f) => {
      const time = Date.parse(f.get("time") + "Z"),
        title = f.get("title").trim();
      if (!title || !Number.isFinite(time)) return false;
      const next = { ...m, title, time };
      const milestones = isNew
        ? [...state.milestones, next]
        : state.milestones.map((x) => (x.id === m.id ? next : x));
      try {
        assertTimeRange({ ...state, milestones });
      } catch (e) {
        $("#formError").textContent = e.message;
        return false;
      }
      state.milestones = milestones;
      ensureRange();
      save();
    },
  );
  if (!isNew)
    $("#deleteMilestone").onclick = () => {
      state.milestones = state.milestones.filter((x) => x.id !== m.id);
      $("#editor").close();
      render();
      save();
    };
}
$("#addMilestone").onclick = () => milestoneEditor();
function categoryEditor(id) {
  const c = state.categories.find((c) => c.id === id);
  showEditor(
    `<h2>${c ? "编辑类别" : "新建类别"}</h2><label>类别名称<input name="name" maxlength="80" required value="${esc(c?.name)}" autofocus></label><label>类别说明<textarea name="description" maxlength="300">${esc(c?.description)}</textarea></label>${c ? '<button type="button" class="danger" id="deleteCategory">删除类别及其任务</button>' : ""}`,
    (f) => {
      if (!f.get("name").trim()) return false;
      if (c) {
        c.name = f.get("name").trim();
        c.description = f.get("description");
      } else
        state.categories.push({
          id: uid(),
          name: f.get("name").trim(),
          description: f.get("description"),
          rows: 1,
        });
      render();
      save();
    },
  );
  if (c)
    $("#deleteCategory").onclick = () => {
      if (confirm("确定删除该类别及其中的所有任务？")) {
        state.tasks = state.tasks.filter((t) => t.categoryId !== c.id);
        state.categories = state.categories.filter((x) => x.id !== c.id);
        pruneDependencies(state);
        $("#editor").close();
        render();
        save();
      }
    };
}
function taskEditor(task, isNew = false) {
  showEditor(
    `<h2>${isNew ? "新建任务" : "编辑任务"}</h2><label>任务名称<input name="title" maxlength="120" required value="${esc(task.title)}" autofocus></label><div class="date-fields"><label>开始时间<input type="datetime-local" name="start" required value="${new Date(task.start).toISOString().slice(0, 16)}"></label><label>结束时间<input type="datetime-local" name="end" required value="${new Date(task.end).toISOString().slice(0, 16)}"></label></div><label>任务内容<textarea name="notes" maxlength="5000" placeholder="记录任务目标、备注或待办事项…">${esc(task.notes)}</textarea></label><p class="form-error" id="formError"></p>${!isNew ? '<button type="button" class="danger" id="deleteTask">删除任务</button>' : ""}`,
    (f) => {
      const start = Date.parse(f.get("start") + "Z"),
        end = Date.parse(f.get("end") + "Z");
      if (
        !f.get("title").trim() ||
        !Number.isFinite(start) ||
        !Number.isFinite(end) ||
        end <= start
      ) {
        $("#formError").textContent =
          "请填写任务名称，并确保结束时间晚于开始时间。";
        return false;
      }
      if (
        Math.max(today, end, ...state.tasks.map((t) => t.end)) -
          Math.min(today, start, ...state.tasks.map((t) => t.start)) >
        30 * 366 * DAY
      ) {
        $("#formError").textContent = "计划与今天的总跨度不能超过 30 年。";
        return false;
      }
      const before = structuredClone(state);
      Object.assign(task, {
        title: f.get("title").trim(),
        notes: f.get("notes"),
        start,
        end,
      });
      if (isNew) state.tasks.push(task);
      try {
        if (!isNew) scheduleChange(state, before, task.id);
        settleRows(state);
      } catch (error) {
        state = before;
        if (!isNew) task = state.tasks.find((t) => t.id === task.id);
        $("#formError").textContent = error.message;
        return false;
      }
      ensureRange();
      save();
    },
  );
  if (!isNew) {
    const related = state.dependencies.filter(
      (e) => e.from === task.id || e.to === task.id,
    );
    if (related.length) {
      $("#deleteTask").insertAdjacentHTML(
        "beforebegin",
        `<div class="dependency-list"><strong>任务依赖 · 完成后开始</strong>${related.map((e) => `<div><span>${esc(state.tasks.find((t) => t.id === e.from).title)} → ${esc(state.tasks.find((t) => t.id === e.to).title)}</span><button type="button" data-unlink-from="${esc(e.from)}" data-unlink-to="${esc(e.to)}">解除</button></div>`).join("")}</div>`,
      );
      $("#editor")
        .querySelectorAll("[data-unlink-from]")
        .forEach(
          (b) =>
            (b.onclick = () => {
              state.dependencies = state.dependencies.filter(
                (e) =>
                  e.from !== b.dataset.unlinkFrom ||
                  e.to !== b.dataset.unlinkTo,
              );
              b.parentElement.remove();
              render();
              save();
            }),
        );
    }
  }
  if (!isNew)
    $("#deleteTask").onclick = () => {
      state.tasks = state.tasks.filter((t) => t.id !== task.id);
      pruneDependencies(state);
      $("#editor").close();
      render();
      save();
    };
}
$("#addCategory").onclick = () => categoryEditor();
$("#canvas").addEventListener("click", (e) => {
  const colorButton = e.target.closest("[data-color]");
  if (colorButton) {
    const palette = document.querySelector(`[data-palette="${colorButton.dataset.color}"]`);
    const rect = colorButton.getBoundingClientRect();
    palette.style.left = Math.max(8, Math.min(rect.left, innerWidth - 218)) + 'px';
    palette.style.top = Math.max(8, Math.min(rect.bottom + 8, innerHeight - 170)) + 'px';
    palette.togglePopover();
    colorButton.setAttribute('aria-expanded', String(palette.matches(':popover-open')));
    return;
  }
  const paletteColor = e.target.closest("[data-palette-color]");
  if (paletteColor) {
    const c = state.categories.find(c => c.id === paletteColor.dataset.paletteColor);
    c.color = paletteColor.dataset.value; render(); save(); return;
  }
  const visibility = e.target.closest('[data-visibility]');
  if (visibility) {
    const c = state.categories.find(c => c.id === visibility.dataset.visibility);
    c.exportVisible = c.exportVisible === false;
    render(); save();
    document.querySelector('[data-visibility="' + c.id + '"]')?.focus();
    return;
  }
  const rowButton = e.target.closest("[data-delete-row]");
  if (rowButton) {
    deleteEmptyRow(
      state,
      rowButton.dataset.rowCategory,
      Number(rowButton.dataset.deleteRow),
    );
    render();
    save();
    return;
  }
  const marker = e.target.closest("[data-milestone]");
  if (marker && performance.now() > suppressMarkerClickUntil)
    milestoneEditor(
      state.milestones.find((m) => m.id === marker.dataset.milestone),
      false,
    );
  const b = e.target.closest(".edit-category");
  if (b) categoryEditor(b.dataset.id);
  if (e.target.closest("#inlineCategory")) categoryEditor();
});
$("#canvas").addEventListener("toggle", e => {
  if (!e.target.matches('.color-palette')) return;
  const button = [...document.querySelectorAll('[data-color]')].find(b => b.dataset.color === e.target.dataset.palette);
  button?.setAttribute('aria-expanded', String(e.newState === 'open'));
}, true);
$("#canvas").addEventListener("change", e => {
  if (!e.target.matches('[data-custom-color]')) return;
  const c = state.categories.find(c => c.id === e.target.dataset.customColor);
  c.color = e.target.value;
  render(); save();
});
$("#canvas").addEventListener("dblclick", (e) => {
  if(e.target.closest("[data-layout],[data-color],[data-visibility]"))return;
  if (e.target.closest(".milestone-track") && !e.target.closest("button")) {
    milestoneEditor({ id: uid(), title: "", time: snap(timeAt(e), true) });
    return;
  }
  const el = e.target.closest(".task");
  if (el) {
    taskEditor(state.tasks.find((t) => t.id === el.dataset.task));
    return;
  }
  const track = e.target.closest(".track");
  if (track) {
    const c = state.categories.find((c) => c.id === track.dataset.category);
    const start = snap(timeAt(e), e.ctrlKey || e.metaKey);
    taskEditor(
      {
        id: uid(),
        categoryId: c.id,
        row: Math.min(
          c.rows - 1,
          Math.max(
            0,
            Math.floor(
              (e.clientY - track.getBoundingClientRect().top - rowPadding()) / rowHeight(),
            ),
          ),
        ),
        title: "",
        notes: "",
        start,
        end: start + WEEK,
      },
      true,
    );
  }
});
let suppressMarkerClickUntil = 0;
$("#viewport").addEventListener("pointerdown", (e) => {
  const grip=e.target.closest('[data-layout]');
  if(e.button===0&&grip){drag={type:'layout',key:grip.dataset.layout,x:e.clientX,y:e.clientY,before:structuredClone(state),original:grip.dataset.layout==='columnWidth'?columnWidth():headerHeight(),moved:false};e.preventDefault();return;}
  const categoryGrip = e.target.closest("[data-reorder]");
  if (e.button === 0 && categoryGrip) {
    drag = { type: "category", id: categoryGrip.dataset.reorder, x: e.clientX, y: e.clientY, moved: false };
    e.preventDefault();
    return;
  }
  const marker = e.target.closest("[data-milestone]");
  if (e.button === 0 && marker) {
    const milestone = state.milestones.find(
      (m) => m.id === marker.dataset.milestone,
    );
    drag = {
      type: "milestone",
      milestone,
      original: milestone.time,
      x: e.clientX,
      y: e.clientY,
      time: timeAt(e),
      moved: false,
    };
    e.preventDefault();
    return;
  }
  if (e.button !== 0 || e.target.closest("button,.category-label,.corner"))
    return;
  const el = e.target.closest(".task");
  if (el) {
    const task = state.tasks.find((t) => t.id === el.dataset.task);
    drag = {
      type: e.shiftKey ? "link" : e.target.dataset.resize || "move",
      before: structuredClone(state),
      task,
      original: { ...task },
      x: e.clientX,
      y: e.clientY,
      time: timeAt(e),
      moved: false,
    };
  } else {
    drag = {
      type: "pan",
      x: e.clientX,
      y: e.clientY,
      scroll: $("#viewport").scrollLeft,
      top: $("#viewport").scrollTop,
    };
  }
  e.preventDefault();
});
document.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const v = $("#viewport");
  if (drag.type === "pan") {
    v.scrollLeft = drag.scroll + drag.x - e.clientX;
    v.scrollTop = drag.top + drag.y - e.clientY;
    return;
  }
  if (
    Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) < 4 &&
    !drag.moved
  )
    return;
  drag.moved = true;
  if (drag.type === "category") {
    document.querySelectorAll(".category").forEach(el => el.classList.remove("drop-before", "drop-after"));
    const bounds = v.getBoundingClientRect();
    drag.target = null;
    if (e.clientX < bounds.left || e.clientX > bounds.right || e.clientY < bounds.top || e.clientY > bounds.bottom) return;
    const categories = [...document.querySelectorAll(".category")];
    const target = categories.find(el => e.clientY < el.getBoundingClientRect().bottom) || categories.at(-1);
    if (target && target.dataset.category !== drag.id) {
      const rect = target.getBoundingClientRect();
      drag.after = e.clientY >= rect.top + rect.height / 2;
      drag.target = target.dataset.category;
      target.classList.add(drag.after ? "drop-after" : "drop-before");
    }
    if (e.clientY > bounds.bottom - 35) v.scrollTop += 16;
    else if (e.clientY < bounds.top + 35) v.scrollTop -= 16;
    return;
  }
  if(drag.type==='layout'){const col=drag.key==='columnWidth';state.settings[drag.key]=Math.round(Math.max(col?140:48,Math.min(col?480:180,drag.original+(col?e.clientX-drag.x:e.clientY-drag.y))));render();return;}
  if (drag.type === "milestone") {
    const next = snap(
      drag.original + timeAt(e) - drag.time,
      e.ctrlKey || e.metaKey,
    );
    const previous = drag.milestone.time;
    drag.milestone.time = next;
    try {
      assertTimeRange(state);
    } catch (error) {
      drag.milestone.time = previous;
      notify(error.message);
    }
    render();
    [...document.querySelectorAll(".milestone")]
      .find((el) => el.dataset.milestone === drag.milestone.id)
      ?.classList.add("dragging");
    if (e.clientX > v.getBoundingClientRect().right - 40) v.scrollLeft += 16;
    else if (e.clientX < v.getBoundingClientRect().left + 250)
      v.scrollLeft -= 16;
    return;
  }
  if (drag.type === "link") {
    document
      .querySelectorAll(".link-target")
      .forEach((el) => el.classList.remove("link-target"));
    const target = document
      .elementFromPoint(e.clientX, e.clientY)
      ?.closest(".task");
    drag.linkTo = target?.dataset.task;
    if (target && drag.linkTo !== drag.task.id)
      target.classList.add("link-target");
    drawOverlay(e);
    return;
  }
  for (const original of drag.before.tasks)
    Object.assign(
      state.tasks.find((t) => t.id === original.id),
      original,
    );
  const free = e.ctrlKey || e.metaKey,
    delta = timeAt(e) - drag.time,
    t = drag.task,
    o = drag.original;
  if (drag.type === "move") {
    t.start = snap(o.start + delta, free);
    t.end = t.start + o.end - o.start;
  } else if (drag.type === "start")
    t.start = Math.min(
      snap(o.start + delta, free),
      o.end - (free ? 60000 : WEEK),
    );
  else
    t.end = Math.max(
      snap(o.end + delta, free),
      o.start + (free ? 60000 : WEEK),
    );
  try {
    scheduleChange(state, drag.before, t.id);
  } catch (error) {
    for (const original of drag.before.tasks)
      Object.assign(
        state.tasks.find((t) => t.id === original.id),
        original,
      );
    notify(error.message);
  }
  render();
  const track = document
    .elementsFromPoint(e.clientX, e.clientY)
    .map((el) => el.closest(".track"))
    .find(Boolean);
  drag.target = null;
  if (track && drag.type === "move") {
    const rect = track.getBoundingClientRect(),
      c = state.categories.find((c) => c.id === track.dataset.category),
      y = e.clientY - rect.top;
    drag.target = {
      category: c.id,
      row: Math.max(0, Math.min(c.rows - 1, Math.floor((y - rowPadding()) / rowHeight()))),
      edge: y < rowPadding()+1 ? "top" : y > rect.height - rowPadding()-1 ? "bottom" : null,
    };
    if (drag.target.edge)
      track.querySelector(".edge." + drag.target.edge).classList.add("visible");
  }
  const node = [...document.querySelectorAll(".task")].find(
    (el) => el.dataset.task === t.id,
  );
  node?.classList.add("dragging");
  if (node && track && drag.target) {
    track.append(node);
    node.style.top = `${rowPadding() + drag.target.row * rowHeight()}px`;
  }
  drawOverlay();
  if (e.clientX > v.getBoundingClientRect().right - 40) v.scrollLeft += 16;
  else if (e.clientX < v.getBoundingClientRect().left + 250) v.scrollLeft -= 16;
});
function finish(cancel = false) {
  if (!drag) return;
  if (drag.type === "category") {
    if (!cancel && drag.moved && drag.target) {
      const category = state.categories.find(c => c.id === drag.id);
      const remaining = state.categories.filter(c => c.id !== drag.id);
      const index = remaining.findIndex(c => c.id === drag.target) + Number(drag.after);
      remaining.splice(index, 0, category);
      state.categories = remaining;
      save();
    }
    drag = null;
    render();
    return;
  }
  if(drag.type==='layout'){if(cancel)state=drag.before;else if(drag.moved)save();drag=null;render();return;}
  if (drag.type === "milestone") {
    if (drag.moved) {
      suppressMarkerClickUntil = performance.now() + 350;
      if (cancel) drag.milestone.time = drag.original;
      else save();
      ensureRange();
    }
    drag = null;
    return;
  }
  if (drag.type === "link") {
    if (!cancel && drag.moved && drag.linkTo) {
      try {
        linkTasks(state, drag.task.id, drag.linkTo);
        assertTimeRange(state);
        settleRows(state);
        save();
        notify("依赖已建立：后项在前项完成后开始");
      } catch (error) {
        state = drag.before;
        notify(error.message);
      }
    }
    drag = null;
    ensureRange();
    return;
  }
  if (drag.task && drag.moved) {
    if (cancel) state = drag.before;
    else if (drag.moved) {
      const target = drag.target;
      placeTask(
        state,
        drag.task,
        target?.category || drag.task.categoryId,
        target?.row ?? drag.task.row,
        target?.edge,
      );
      settleRows(state);
      save();
    }
    ensureRange();
  }
  drag = null;
}
document.addEventListener("pointerup", () => finish());
document.addEventListener("pointercancel", () => finish(true));
window.addEventListener("blur", () => finish(true));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") finish(true);
});
function closeExportMenu() {
  $("#exportMenu").hidden = true;
  $("#export").setAttribute("aria-expanded", "false");
}
$("#export").onclick = () => {
  const menu = $("#exportMenu");
  menu.hidden = !menu.hidden;
  $("#export").setAttribute("aria-expanded", String(!menu.hidden));
};
$("#export").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") {
    e.preventDefault();
    $("#exportMenu").hidden = false;
    $("#export").setAttribute("aria-expanded", "true");
    $("#exportMenu button").focus();
  }
});
$("#exportMenu").addEventListener("keydown", (e) => {
  const items = [...$("#exportMenu").querySelectorAll("button")],
    index = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    items[
      (index + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length
    ].focus();
  }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".export-control")) closeExportMenu();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#exportMenu").hidden) {
    closeExportMenu();
    $("#export").focus();
  }
});
function confirmConflictExport(conflicts) {
  const titles = new Map(state.tasks.map(t => [t.id, t.title]));
  const dialog = document.createElement("dialog");
  dialog.className = "export-warning sketch-dialog";
  dialog.setAttribute("aria-labelledby", "exportWarningTitle");
  dialog.setAttribute("aria-describedby", "exportWarningDescription");
  dialog.innerHTML = `<form method="dialog">
    <div class="sketch-eyebrow"><span aria-hidden="true">!</span> 导出前，再看一眼</div>
    <h2 id="exportWarningTitle">有几条时间线打结了</h2>
    <p id="exportWarningDescription">发现 <b>${conflicts.length} 处依赖矛盾</b>：前项的结束时间晚于后项的开始时间。</p>
    <ul class="conflict-notes">${conflicts.map(e => `<li><span>${esc(titles.get(e.from))}</span><span class="conflict-arrow" aria-label="依赖于后续任务">→</span><span>${esc(titles.get(e.to))}</span></li>`).join("")}</ul>
    <p class="sketch-hint">可以返回调整时间，或勾选“锁定依赖关系”自动修正。继续导出会保留当前安排。</p>
    <div class="dialog-actions"><button value="cancel" autofocus>取消导出</button><button value="continue" class="primary">继续导出 <span aria-hidden="true">↗</span></button></div>
  </form>`;
  document.body.append(dialog);
  return new Promise(resolve => {
    dialog.addEventListener("close", () => {
      const accepted = dialog.returnValue === "continue";
      dialog.remove();
      $("#export").focus();
      resolve(accepted);
    }, { once: true });
    dialog.showModal();
  });
}
$("#exportMenu").onclick = async (e) => {
  const button = e.target.closest("[data-export]");
  if (!button) return;
  const format = button.dataset.export;
  closeExportMenu();
  const conflicts = dependencyConflicts(format === "json" ? state : imageState(state));
  if (conflicts.length) {
    if (!await confirmConflictExport(conflicts)) return;
  }
  $("#export").disabled = true;
  $("#export").textContent = "正在导出…";
  try {
    if (format === "json")
      download(
        new Blob([JSON.stringify({ ...state, version: 2 }, null, 2)], {
          type: "application/json",
        }),
        `${safeName(state.name)}-${date(today)}.json`,
      );
    else await exportImage(state, format);
    notify("导出完成");
  } catch (error) {
    notify("导出失败：" + error.message);
  } finally {
    $("#export").disabled = false;
    $("#export").textContent = "↧ 导出项目 ▾";
    $("#export").focus();
  }
};
$("#import").onclick = () => $("#file").click();
$("#file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    if (file.size > 5e6) throw Error("文件不能超过 5 MB");
    const next = validate(JSON.parse(await file.text()));
    createProject(
      typeof next.name === "string" && next.name.trim()
        ? next.name
        : file.name.replace(/\.json$/i, ""),
      next,
    );
  } catch (error) {
    alert("导入失败：" + error.message);
  }
  e.target.value = "";
};
function safeName(name) {
  return (
    name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").slice(0, 100) || "gantt-project"
  );
}
function applyDisplaySettings() {
  document.documentElement.style.setProperty('--column-width',columnWidth()+'px');
  document.documentElement.style.setProperty('--header-height',headerHeight()+'px');
  state.settings ??= { showTaskDates: true, fontSize: 14 };
  $('#compactMode').checked=!!state.settings.compact;
  document.body.classList.toggle('compact-mode',!!state.settings.compact);
  const compact = !!state.settings.compact;
  const header = $("main > header");
  const title = $(".intro > div:first-child");
  const actions = $(".intro .actions");
  const crumb = $(".crumb");
  const saved = $("#saved");
  if (compact) {
    title.prepend(crumb);
    actions.prepend(saved);
  } else {
    header.append(crumb, saved);
  }
  $('#projectStats').hidden=!!state.settings.statsCollapsed;
  $('#toggleStats').setAttribute('aria-expanded',String(!state.settings.statsCollapsed));
  $('#toggleStats').textContent=state.settings.statsCollapsed?'▸ 项目概览':'▾ 项目概览';
  document.body.classList.toggle('stats-collapsed',!!state.settings.statsCollapsed);
  $("#showTaskDates").checked = state.settings.showTaskDates;
  $("#lockDependencies").checked = !!state.settings.lockDependencies;
  $("#hideOverflowText").checked = !!state.settings.hideOverflowText;
  document.body.classList.toggle("hide-overflow-text", !!state.settings.hideOverflowText);
  $("#fontSize").value = String(state.settings.fontSize);
  document.documentElement.style.setProperty(
    "--text-scale",
    state.settings.fontSize / 14,
  );
  document.body.classList.toggle(
    "hide-task-dates",
    !state.settings.showTaskDates,
  );
}
$('#toggleStats').onclick=()=>{state.settings.statsCollapsed=!state.settings.statsCollapsed;render();save();};
$('#compactMode').onchange=e=>{state.settings.compact=e.target.checked;if(e.target.checked)state.settings.showTaskDates=false;render();save();};
$("#showTaskDates").onchange = (e) => {
  state.settings.showTaskDates = e.target.checked;
  render();
  save();
};
$("#lockDependencies").onchange = e => {
  const before = structuredClone(state);
  try {
    state.settings.lockDependencies = e.target.checked;
    if (e.target.checked) {
      enforceDependencies(state);
      assertTimeRange(state);
      settleRows(state);
    }
    render();
    save();
  } catch (error) {
    state = before;
    render();
    notify(error.message);
  }
};
$("#hideOverflowText").onchange = e => {
  state.settings.hideOverflowText = e.target.checked;
  render();
  save();
};
$("#fontSize").onchange = (e) => {
  state.settings.fontSize = Number(e.target.value);
  render();
  save();
};
function renderProjectList() {
  if (!library) return;
  $("#projectList").innerHTML = library.projects
    .slice()
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .map(
      (p) =>
        `<button class="project-entry ${p.id === library.activeId ? "selected" : ""}" data-open-project="${esc(p.id)}"><strong>${esc(p.name)}</strong><small>${p.data.tasks.length} 个任务 · ${new Date(p.updatedAt).toLocaleString("zh-CN")}</small>${p.id === library.activeId ? "<span>当前项目</span>" : ""}</button>`,
    )
    .join("");
}
function closeProjects() {
  $("#projectPanel").hidden = true;
  $("#projectBackdrop").hidden = true;
  $("#projectsButton").setAttribute("aria-expanded", "false");
}
$("#projectsButton").onclick = () => {
  const open = $("#projectPanel").hidden;
  $("#projectPanel").hidden = !open;
  $("#projectBackdrop").hidden = !open;
  $("#projectsButton").setAttribute("aria-expanded", String(open));
  renderProjectList();
};
$("#closeProjects").onclick = closeProjects;
$("#projectBackdrop").onclick = closeProjects;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeProjects();
});
function activateProject() {
  state = projectState(library);
  history.entry(library.activeId, state);
  updateHistoryButtons();
  state.settings ??= { showTaskDates: true, fontSize: 14 };
  scale = 112 / WEEK;
  origin = base - 12 * WEEK;
  ensureRange();
  goToday();
  $("#viewport").scrollTop = 0;
  renderProjectList();
  closeProjects();
}
$("#projectList").onclick = (e) => {
  const b = e.target.closest("[data-open-project]");
  if (!b || b.dataset.openProject === library.activeId) return;
  if (!save()) {
    notify("当前项目保存失败，请先导出备份");
    return;
  }
  const next = { ...library, activeId: b.dataset.openProject };
  try {
    localStorage.setItem(LIBRARY_KEY, JSON.stringify(next));
    library = next;
    activateProject();
  } catch {
    notify("切换失败：本地存储不可用");
  }
};
function createProject(name, data) {
  if (!save()) {
    notify("当前项目保存失败，请先导出备份");
    return;
  }
  const id = uid(),
    project = {
      id,
      name: name || `未命名项目 ${library.projects.length + 1}`,
      updatedAt: Date.now(),
      data: data || {
        categories: [{ id: uid(), name: "任务类别", description: "", rows: 1 }],
        tasks: [],
        milestones: [],
        dependencies: [],
        settings: { showTaskDates: true, fontSize: 14 },
      },
    };
  const next = {
    ...library,
    activeId: id,
    projects: [...library.projects, project],
  };
  try {
    localStorage.setItem(LIBRARY_KEY, JSON.stringify(next));
    library = next;
    activateProject();
    save();
  } catch {
    notify("新建失败：本地存储不可用");
  }
}
$("#newProject").onclick = () => createProject();
$("#panelNewProject").onclick = () => createProject();
function renameProject() {
  const input = $("#projectNameInput");
  input.value = state.name;
  $("#projectTitle").hidden = true;
  input.hidden = false;
  input.focus();
  input.select();
}
$("#projectTitle").ondblclick = renameProject;
$("#projectTitle").onkeydown = (e) => {
  if (e.key === "Enter" || e.key === "F2") {
    e.preventDefault();
    renameProject();
  }
};
function finishRename(cancel = false) {
  const input = $("#projectNameInput");
  if (input.hidden) return;
  const name = input.value.trim();
  if (!cancel && name) {
    state.name = name;
    save();
  }
  input.hidden = true;
  $("#projectTitle").hidden = false;
  $("#projectTitle").textContent = state.name;
}
$("#projectNameInput").onblur = () => finishRename();
$("#projectNameInput").onkeydown = (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    finishRename();
  } else if (e.key === "Escape") {
    e.stopPropagation();
    finishRename(true);
  }
};
function updateHistoryButtons() {
  const status = history.status(library?.activeId);
  $("#undo").disabled = !status.undo;
  $("#redo").disabled = !status.redo;
}
function travelHistory(direction) {
  if ($("#editor").open || !$("#projectNameInput").hidden) return;
  finish(true);
  const previous = state,
    next = history[direction](library?.activeId);
  if (!next) return;
  state = next;
  if (!save(false)) {
    history[direction === "undo" ? "redo" : "undo"](library.activeId);
    state = previous;
    notify("保存失败，操作未执行");
  }
  ensureRange();
  updateHistoryButtons();
}
$("#undo").onclick = () => travelHistory("undo");
$("#redo").onclick = () => travelHistory("redo");
function textEditing() {
  const el = document.activeElement;
  return el?.matches('textarea,input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="button"])') || el?.isContentEditable;
}
document.addEventListener("keydown", (e) => {
  if (
    (e.ctrlKey || e.metaKey) &&
    !e.altKey &&
    e.key.toLowerCase() === "z" &&
    !textEditing() &&
    !$("#editor").open
  ) {
    e.preventDefault();
    travelHistory(e.shiftKey ? "redo" : "undo");
  }
});
window.addEventListener("gantt-history", (e) => {
  if (textEditing()) document.execCommand(e.detail);
  else travelHistory(e.detail);
});
render();
ensureRange();
goToday();
if (saveError)
  $("#saved").textContent =
    "项目库无法读取，原始数据已保留，请导出当前内容备份";
else save();

$('#canvas').addEventListener('keydown',e=>{const grip=e.target.closest('[data-layout]');if(!grip)return;const col=grip.dataset.layout==='columnWidth',direction=col?{ArrowLeft:-1,ArrowRight:1}:{ArrowUp:-1,ArrowDown:1};if(!direction[e.key])return;e.preventDefault();state.settings[grip.dataset.layout]=Math.max(col?140:48,Math.min(col?480:180,(col?columnWidth():headerHeight())+direction[e.key]*4));render();save();document.querySelector(`[data-layout="${grip.dataset.layout}"]`)?.focus();});


$("#canvas").addEventListener("keydown", e => {
  const grip = e.target.closest("[data-reorder]");
  if (!grip || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
  e.preventDefault();
  const from = state.categories.findIndex(c => c.id === grip.dataset.reorder);
  const to = from + (e.key === "ArrowUp" ? -1 : 1);
  if (to < 0 || to >= state.categories.length) return;
  const [category] = state.categories.splice(from, 1);
  state.categories.splice(to, 0, category);
  render();
  save();
  [...document.querySelectorAll("[data-reorder]")].find(el => el.dataset.reorder === category.id)?.focus();
});
