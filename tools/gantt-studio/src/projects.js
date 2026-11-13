import { validate } from "./model.js";
export const LIBRARY_KEY = "gantt-studio-projects-v1";
export function loadLibrary(storage, initial) {
  const raw = storage.getItem(LIBRARY_KEY);
  if (raw) {
    const value = JSON.parse(raw);
    if (!value || !Array.isArray(value.projects) || !value.projects.length)
      throw Error("项目库数据无效");
    const ids = new Set();
    for (const p of value.projects) {
      if (
        !p ||
        typeof p.id !== "string" ||
        ids.has(p.id) ||
        typeof p.name !== "string" ||
        !p.name.trim()
      )
        throw Error("项目数据无效");
      ids.add(p.id);
      p.data = validate(p.data);
    }
    if (!ids.has(value.activeId)) value.activeId = value.projects[0].id;
    return value;
  }
  const legacy = storage.getItem("gantt-studio-v1");
  const data = legacy ? validate(JSON.parse(legacy)) : validate(initial);
  const project = {
    id: crypto.randomUUID(),
    name:
      typeof data.name === "string" && data.name.trim()
        ? data.name
        : "项目时间线",
    updatedAt: Date.now(),
    data,
  };
  return { version: 1, activeId: project.id, projects: [project] };
}
export function persistProject(storage, library, state) {
  const next = structuredClone(library),
    project = next.projects.find((p) => p.id === next.activeId);
  project.data = structuredClone(state);
  project.name = state.name || project.name;
  project.updatedAt = Date.now();
  storage.setItem(LIBRARY_KEY, JSON.stringify(next));
  // Retain a compatible current-project snapshot for older app versions.
  try {
    storage.setItem("gantt-studio-v1", JSON.stringify(state));
  } catch {}
  return next;
}
export function projectState(library) {
  const p = library.projects.find((p) => p.id === library.activeId);
  return { ...structuredClone(p.data), name: p.name };
}
