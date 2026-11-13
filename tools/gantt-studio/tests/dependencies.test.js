import test from "node:test";
import assert from "node:assert/strict";
import {
  DAY,
  linkTasks,
  scheduleChange,
  deleteEmptyRow,
  validate,
  pruneDependencies,
} from "../src/model.js";
import { projectSVG } from "../src/export.js";
const base = Date.UTC(2026, 9, 26);
function sample() {
  return {
    categories: [{ id: "c", name: "研发", rows: 4 }],
    tasks: [
      ["a", 0, 2],
      ["b", 3, 5],
      ["c", 6, 8],
    ].map(([id, start, end], row) => ({
      id,
      title: id,
      categoryId: "c",
      row,
      start: base + start * DAY,
      end: base + end * DAY,
    })),
    dependencies: [],
    milestones: [],
    settings:{showTaskDates:true,fontSize:14,lockDependencies:true},
  };
}
test("连接后推迟不满足约束的后项，并拒绝循环、自连接、重复连接", () => {
  const s = sample();
  linkTasks(s, "a", "b");
  linkTasks(s, "b", "c");
  assert.throws(() => linkTasks(s, "c", "a"), /循环/);
  assert.throws(() => linkTasks(s, "a", "a"));
  assert.throws(() => linkTasks(s, "a", "b"), /重复/);
  assert.equal(s.dependencies.length, 2);
  const x = sample();
  linkTasks(x, "c", "a");
  assert.equal(x.tasks[0].start, x.tasks[2].end);
  assert.equal(x.tasks[0].end - x.tasks[0].start, 2 * DAY);
});
test("前移、后移及边框调整沿多级依赖传播并保留时间间隔", () => {
  for (const delta of [-DAY, 4 * DAY]) {
    const s = sample();
    linkTasks(s, "a", "b");
    linkTasks(s, "b", "c");
    const before = structuredClone(s);
    s.tasks[0].end += delta;
    scheduleChange(s, before, "a");
    assert.equal(s.tasks[1].start, before.tasks[1].start + delta);
    assert.equal(s.tasks[2].end, before.tasks[2].end + delta);
    assert.equal(s.tasks[2].end - s.tasks[2].start, 2 * DAY);
  }
});
test("多个前置任务取最晚约束，直接移动后项受限，菱形依赖只传播一次", () => {
  const s = sample();
  s.tasks.push({
    id: "d",
    title: "d",
    categoryId: "c",
    row: 3,
    start: base + 9 * DAY,
    end: base + 10 * DAY,
  });
  linkTasks(s, "a", "b");
  linkTasks(s, "a", "c");
  linkTasks(s, "b", "d");
  linkTasks(s, "c", "d");
  const before = structuredClone(s);
  s.tasks[0].end += 3 * DAY;
  scheduleChange(s, before, "a");
  assert.equal(s.tasks[3].start, before.tasks[3].start + 3 * DAY);
  const prev = structuredClone(s);
  s.tasks[1].start = base;
  s.tasks[1].end = base + DAY;
  scheduleChange(s, prev, "b");
  assert.equal(s.tasks[1].start, s.tasks[0].end);
  assert.equal(s.tasks[1].end - s.tasks[1].start, DAY);
});
test("删除空行移动行索引但不改变时间与依赖，不允许删除有任务的行", () => {
  const s = sample();
  s.tasks[2].row = 3;
  linkTasks(s, "a", "b");
  const before = structuredClone(s);
  deleteEmptyRow(s, "c", 2);
  assert.equal(s.categories[0].rows, 3);
  assert.equal(s.tasks[2].row, 2);
  assert.equal(s.tasks[2].start, before.tasks[2].start);
  assert.deepEqual(s.dependencies, before.dependencies);
  assert.throws(() => deleteEmptyRow(s, "c", 0));
  s.tasks = [];
  deleteEmptyRow(s, "c", 1);
  deleteEmptyRow(s, "c", 0);
  assert.throws(() => deleteEmptyRow(s, "c", 0), /至少/);
});
test("旧项目兼容，关键点与依赖往返导入，删除任务清理关系", () => {
  const s = sample();
  s.milestones = [{ id: "m", title: "发布", time: base + DAY }];
  linkTasks(s, "a", "b");
  assert.deepEqual(validate(JSON.parse(JSON.stringify(s))), s);
  const old = { ...s };
  delete old.dependencies;
  delete old.milestones;
  assert.deepEqual(validate(old).dependencies, []);
  assert.throws(() =>
    validate({ ...s, dependencies: [{ from: "a", to: "absent" }] }),
  );
  assert.throws(() =>
    validate({ ...s, milestones: [{ id: "m", title: "错误", time: NaN }] }),
  );
  s.tasks = s.tasks.filter((t) => t.id !== "a");
  pruneDependencies(s);
  assert.equal(s.dependencies.length, 0);
});
test("完整 SVG 导出包含关键点、虚线曲线，转义文本且无外部资源", () => {
  const s = sample();
  linkTasks(s, "a", "b");
  s.milestones = [{ id: "m", title: "发布<&", time: base + DAY }];
  const { svg, width, height } = projectSVG(s);
  assert.match(svg, /发布&lt;&amp;/);
  assert.match(svg, /stroke-dasharray="5 5"/);
  assert.match(svg, / C /);
  assert.doesNotMatch(svg, /<foreignObject|<script|<image/);
  assert.equal(width, 904); // Two occupied weeks plus one spare on each side.
  assert.ok(height > 400);
  assert.match(projectSVG({ tasks: [], categories: [] }).svg, /项目时间线/);
});
