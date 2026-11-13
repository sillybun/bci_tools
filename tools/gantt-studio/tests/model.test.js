import test from "node:test";
import assert from "node:assert/strict";
import { WEEK, DAY, monday, snap, placeTask, validate } from "../src/model.js";
const start = Date.UTC(2026, 9, 26);
test("周一边界及 Ctrl 自由时间精度", () => {
  assert.equal(monday(start + 3 * DAY), start);
  assert.equal(snap(start + 2 * DAY), start);
  assert.equal(snap(start + 5 * DAY), start + WEEK);
  assert.equal(snap(start + 123456, true), start + 120000);
});
test("相接任务无需新增行，重叠任务插入行且移动下方任务", () => {
  const a = { id: "a", categoryId: "c", row: 0, start, end: start + WEEK },
    b = { id: "b", categoryId: "c", row: 1, start, end: start + WEEK },
    t = {
      id: "t",
      categoryId: "c",
      row: 0,
      start: start + WEEK,
      end: start + 2 * WEEK,
    };
  const s = { categories: [{ id: "c", rows: 2 }], tasks: [a, b, t] };
  placeTask(s, t, "c", 0);
  assert.equal(s.categories[0].rows, 2);
  t.start = start;
  placeTask(s, t, "c", 0);
  assert.equal(t.row, 1);
  assert.equal(b.row, 2);
  assert.equal(s.categories[0].rows, 3);
});
test("顶部和底部插入行与跨类别移动", () => {
  const t = { id: "a", categoryId: "a", row: 0, start, end: start + WEEK },
    b = { id: "b", categoryId: "b", row: 0, start, end: start + WEEK };
  const s = {
    categories: [
      { id: "a", rows: 1 },
      { id: "b", rows: 1 },
    ],
    tasks: [t, b],
  };
  placeTask(s, t, "b", 0, "top");
  assert.equal(t.row, 0);
  assert.equal(b.row, 1);
  assert.equal(t.categoryId, "b");
  placeTask(s, t, "b", 0, "bottom");
  assert.equal(t.row, 2);
  assert.equal(s.categories[1].rows, 3);
});
test("拒绝损坏的导入并修复已有重叠", () => {
  assert.throws(() => validate({ categories: [], tasks: [{}] }));
  const t = {
    id: "a",
    categoryId: "c",
    row: 0,
    title: "A",
    start,
    end: start + WEEK,
  };
  const s = validate({
    categories: [{ id: "c", name: "C", rows: 1 }],
    tasks: [t, { ...t, id: "b" }],
  });
  assert.notEqual(s.tasks[0].row, s.tasks[1].row);
  assert.throws(() =>
    validate({
      categories: [{ id: "c", name: "C", rows: 1 }],
      tasks: [{ ...t, end: start }],
    }),
  );
});
