export const CATEGORY_COLORS = ["#6961d9", "#299c91", "#cf9250", "#568acb", "#bf6f9c"];
export function categoryColor(category, index = 0) {
  return category?.color ?? CATEGORY_COLORS[index % CATEGORY_COLORS.length];
}
export function imageState(state) {
  const categories = state.categories.map((c, i) => ({...c, color: categoryColor(c, i)})).filter(c => c.exportVisible !== false);
  const ids = new Set(categories.map(c => c.id));
  const tasks = state.tasks.filter(t => ids.has(t.categoryId));
  const taskIds = new Set(tasks.map(t => t.id));
  return {...state, categories, tasks, dependencies: (state.dependencies ?? []).filter(e => taskIds.has(e.from) && taskIds.has(e.to))};
}
export const DAY = 86400000,
  WEEK = 7 * DAY;
export function monday(time) {
  const d = new Date(time);
  return Date.UTC(
    d.getUTCFullYear(),
    d.getUTCMonth(),
    d.getUTCDate() - ((d.getUTCDay() + 6) % 7),
  );
}
export function snap(time, free = false) {
  return free ? Math.round(time / 60000) * 60000 : monday(time + WEEK / 2);
}
export function overlaps(a, b) {
  return a.start < b.end && b.start < a.end;
}
export function placeTask(state, task, categoryId, row, edge) {
  const category = state.categories.find((c) => c.id === categoryId);
  if (!category) return;
  if (edge) {
    row = edge === "top" ? 0 : category.rows;
    state.tasks
      .filter(
        (t) => t.id !== task.id && t.categoryId === categoryId && t.row >= row,
      )
      .forEach((t) => t.row++);
    category.rows++;
  } else {
    row = Math.max(0, Math.min(category.rows - 1, row));
    if (
      state.tasks.some(
        (t) =>
          t.id !== task.id &&
          t.categoryId === categoryId &&
          t.row === row &&
          overlaps(t, task),
      )
    ) {
      row++;
      state.tasks
        .filter(
          (t) =>
            t.id !== task.id && t.categoryId === categoryId && t.row >= row,
        )
        .forEach((t) => t.row++);
      category.rows++;
    }
  }
  task.categoryId = categoryId;
  task.row = row;
}
export function validate(data) {
  if (
    !data ||
    !Array.isArray(data.categories) ||
    !Array.isArray(data.tasks) ||
    data.categories.length > 100 ||
    data.tasks.length > 5000
  )
    throw Error("项目格式不正确或数据量过大");
  const ids = new Set();
  for (const c of data.categories) {
    if (
      typeof c.id !== "string" ||
      ids.has(c.id) ||
      typeof c.name !== "string" ||
      !Number.isInteger(c.rows) ||
      c.rows < 1 ||
      c.rows > 10000 ||
      (c.color !== undefined && !/^#[0-9a-f]{6}$/i.test(c.color)) ||
      (c.exportVisible !== undefined && typeof c.exportVisible !== "boolean")
    )
      throw Error("类别数据无效");
    ids.add(c.id);
  }
  const tids = new Set();
  for (const t of data.tasks) {
    const c = data.categories.find((c) => c.id === t.categoryId);
    if (
      !c ||
      typeof t.id !== "string" ||
      tids.has(t.id) ||
      typeof t.title !== "string" ||
      !Number.isFinite(t.start) ||
      !Number.isFinite(t.end) ||
      Math.abs(t.start) > 8e15 ||
      Math.abs(t.end) > 8e15 ||
      t.end <= t.start ||
      !Number.isInteger(t.row) ||
      t.row < 0 ||
      t.row >= c.rows
    )
      throw Error("任务数据无效");
    tids.add(t.id);
  }
  const dates = data.tasks.flatMap((t) => [t.start, t.end]);
  if (
    Math.max(Date.now(), ...dates) - Math.min(Date.now(), ...dates) >
    30 * 366 * DAY
  )
    throw Error("时间范围须位于今天前后合计 30 年内");
  const result = structuredClone(data);
  result.settings ??= { showTaskDates: true, fontSize: 14 };
  if (
    !result.settings ||
    typeof result.settings.showTaskDates !== "boolean" ||
    ![12, 14, 16, 18].includes(result.settings.fontSize) ||
    (result.settings.hideOverflowText!==undefined && typeof result.settings.hideOverflowText!=="boolean") ||
    (result.settings.lockDependencies!==undefined && typeof result.settings.lockDependencies!=="boolean") ||
    (result.settings.compact!==undefined && typeof result.settings.compact!=="boolean") ||
    (result.settings.statsCollapsed!==undefined && typeof result.settings.statsCollapsed!=="boolean")
  )
    throw Error("显示设置无效");
  if (
    result.name !== undefined &&
    (typeof result.name !== "string" ||
      !result.name.trim() ||
      result.name.length > 100)
  )
    throw Error("项目名称无效");
  for(const [key,min,max] of [['columnWidth',140,480],['headerHeight',48,180]])if(result.settings[key]!==undefined&&(!Number.isFinite(result.settings[key])||result.settings[key]<min||result.settings[key]>max))throw Error('布局尺寸无效');
  result.dependencies ??= [];
  result.milestones ??= [];
  if (
    !Array.isArray(result.dependencies) ||
    result.dependencies.length > 20000 ||
    !Array.isArray(result.milestones) ||
    result.milestones.length > 1000
  )
    throw Error("关键点或依赖数据无效");
  const mids = new Set();
  for (const m of result.milestones) {
    if (
      !m ||
      typeof m.id !== "string" ||
      mids.has(m.id) ||
      typeof m.title !== "string" ||
      !m.title.trim() ||
      m.title.length > 120 ||
      !Number.isFinite(m.time)
    )
      throw Error("关键点数据无效");
    mids.add(m.id);
  }
  dependencyOrder(result);
  if (result.settings.lockDependencies) enforceDependencies(result);
  assertTimeRange(result);
  settleRows(result);
  return result;
}

// Dependencies are finish-to-start edges in a directed acyclic graph.
export function dependencyOrder(state) {
  const tasks = new Map(state.tasks.map((t) => [t.id, t]));
  const incoming = new Map(state.tasks.map((t) => [t.id, 0]));
  const outgoing = new Map(state.tasks.map((t) => [t.id, []]));
  const seen = new Set();
  for (const edge of state.dependencies ?? []) {
    if (
      !edge ||
      !tasks.has(edge.from) ||
      !tasks.has(edge.to) ||
      edge.from === edge.to
    )
      throw Error("依赖关系指向无效任务");
    const key = JSON.stringify([edge.from, edge.to]);
    if (seen.has(key)) throw Error("不能重复连接相同任务");
    seen.add(key);
    outgoing.get(edge.from).push(edge.to);
    incoming.set(edge.to, incoming.get(edge.to) + 1);
  }
  const queue = [...incoming].filter(([, n]) => n === 0).map(([id]) => id);
  for (let i = 0; i < queue.length; i++) {
    for (const id of outgoing.get(queue[i])) {
      incoming.set(id, incoming.get(id) - 1);
      if (incoming.get(id) === 0) queue.push(id);
    }
  }
  if (queue.length !== tasks.size) throw Error("不能建立循环依赖");
  return queue;
}

export function linkTasks(state, from, to) {
  const edge = { from, to };
  const next = {
    ...state,
    dependencies: [...(state.dependencies ?? []), edge],
  };
  dependencyOrder(next);
  state.dependencies = next.dependencies;
  if (state.settings?.lockDependencies) enforceDependencies(state);
}

export function enforceDependencies(state) {
  const tasks = new Map(state.tasks.map((t) => [t.id, t]));
  for (const id of dependencyOrder(state)) {
    const t = tasks.get(id);
    const earliest = Math.max(
      -Infinity,
      ...(state.dependencies ?? [])
        .filter((e) => e.to === id)
        .map((e) => tasks.get(e.from).end),
    );
    if (t.start < earliest) {
      t.end += earliest - t.start;
      t.start = earliest;
    }
  }
}

// Recompute from the gesture snapshot, never from the previous pointer frame.
// This prevents rounding drift and makes Escape restore the entire chain.
export function dependencyConflicts(state) {
  const tasks = new Map(state.tasks.map(t => [t.id, t]));
  return (state.dependencies ?? []).filter(e => tasks.get(e.from)?.end > tasks.get(e.to)?.start);
}

export function scheduleChange(state, before, editedId) {
  if (!state.settings?.lockDependencies) {
    assertTimeRange(state);
    return;
  }
  const tasks = new Map(state.tasks.map((t) => [t.id, t]));
  const old = new Map(before.tasks.map((t) => [t.id, t]));
  const affected = new Set([editedId]);
  for (const id of dependencyOrder(state)) {
    const t = tasks.get(id),
      incoming = (state.dependencies ?? []).filter((e) => e.to === id);
    if (id === editedId) {
      const earliest = Math.max(
        -Infinity,
        ...incoming.map((e) => tasks.get(e.from).end),
      );
      if (t.start < earliest) {
        t.end += earliest - t.start;
        t.start = earliest;
      }
    } else if (incoming.some((e) => affected.has(e.from))) {
      affected.add(id);
      const start = Math.max(
        ...incoming.map(
          (e) =>
            tasks.get(e.from).end +
            Math.max(0, old.get(id).start - old.get(e.from).end),
        ),
      );
      t.start = start;
      t.end = start + old.get(id).end - old.get(id).start;
    }
  }
  assertTimeRange(state);
}

export function assertTimeRange(state) {
  const times = [
    Date.now(),
    ...state.tasks.flatMap((t) => [t.start, t.end]),
    ...(state.milestones ?? []).map((m) => m.time),
  ];
  if (
    times.some((t) => !Number.isFinite(t)) ||
    Math.max(...times) - Math.min(...times) > 30 * 366 * DAY
  )
    throw Error("计划与今天的总跨度不能超过 30 年");
}

export function settleRows(state) {
  for (const t of state.tasks) placeTask(state, t, t.categoryId, t.row);
}

export function pruneDependencies(state) {
  const ids = new Set(state.tasks.map((t) => t.id));
  state.dependencies = (state.dependencies ?? []).filter(
    (e) => ids.has(e.from) && ids.has(e.to),
  );
}

export function deleteEmptyRow(state, categoryId, row) {
  const category = state.categories.find((c) => c.id === categoryId);
  if (!category || !Number.isInteger(row) || row < 0 || row >= category.rows)
    throw Error("行不存在");
  if (category.rows <= 1) throw Error("每个类别至少保留一行用于新建任务");
  if (state.tasks.some((t) => t.categoryId === categoryId && t.row === row))
    throw Error("只能删除空行");
  state.tasks
    .filter((t) => t.categoryId === categoryId && t.row > row)
    .forEach((t) => t.row--);
  category.rows--;
}
