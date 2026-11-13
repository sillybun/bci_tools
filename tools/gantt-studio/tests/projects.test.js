import test from "node:test";
import assert from "node:assert/strict";
import {
  loadLibrary,
  persistProject,
  projectState,
  LIBRARY_KEY,
} from "../src/projects.js";
const data = { categories: [], tasks: [], dependencies: [], milestones: [] };
function storage() {
  const values = new Map();
  return {
    getItem: (k) => values.get(k) ?? null,
    setItem: (k, v) => values.set(k, v),
  };
}
test("旧版迁移保存并重新读取，名称和设置完整保留", () => {
  const s = storage();
  s.setItem("gantt-studio-v1", JSON.stringify(data));
  const library = loadLibrary(s, data);
  const state = projectState(library);
  state.name = "新名称";
  state.settings = { showTaskDates: false, fontSize: 18 };
  persistProject(s, library, state);
  const restored = loadLibrary(s, data);
  assert.equal(projectState(restored).name, "新名称");
  assert.deepEqual(projectState(restored).settings, state.settings);
});
test("存储失败不改变项目库，损坏库不会覆盖原文", () => {
  const s = storage(),
    library = loadLibrary(s, data),
    before = structuredClone(library);
  assert.throws(() =>
    persistProject(
      {
        setItem() {
          throw Error("quota");
        },
      },
      library,
      { ...data, name: "未保存" },
    ),
  );
  assert.deepEqual(library, before);
  s.setItem(LIBRARY_KEY, "invalid");
  assert.throws(() => loadLibrary(s, data));
  assert.equal(s.getItem(LIBRARY_KEY), "invalid");
});
