import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
const DAY = 86400000,
  base = Date.UTC(2026, 9, 26);
const data = () => ({
  categories: [{ id: "c", name: "项目", description: "联动测试", rows: 4 }],
  tasks: [
    {
      id: "a",
      title: "设计",
      categoryId: "c",
      row: 0,
      start: base,
      end: base + 7 * DAY,
    },
    {
      id: "b",
      title: "开发",
      categoryId: "c",
      row: 1,
      start: base + 7 * DAY,
      end: base + 14 * DAY,
    },
    {
      id: "c",
      title: "测试",
      categoryId: "c",
      row: 2,
      start: base + 14 * DAY,
      end: base + 21 * DAY,
    },
  ],
  dependencies: [],
  milestones: [],
});
async function seed(page, state = data()) {
  await page.addInitScript(
    (s) => localStorage.setItem("gantt-studio-v1", JSON.stringify(s)),
    state,
  );
  await page.goto("/");
}
const read = (page) =>
  page.evaluate(() => JSON.parse(localStorage.getItem("gantt-studio-v1")));
async function link(page, from, to) {
  const a = await page.locator(`[data-task="${from}"]`).boundingBox(),
    b = await page.locator(`[data-task="${to}"]`).boundingBox();
  await page.keyboard.down("Shift");
  await page.mouse.move(a.x + 30, a.y + 20);
  await page.mouse.down();
  await page.mouse.move(b.x + 30, b.y + 20, { steps: 8 });
  await expect(page.locator(".dependency-line.preview")).toHaveCount(1);
  await page.mouse.up();
  await page.keyboard.up("Shift");
}
test("Shift 连接、多级联动、循环保护、取消恢复、解除依赖", async ({ page }) => {
  await seed(page);
  await page.locator("#lockDependencies").check();
  await link(page, "a", "b");
  await link(page, "b", "c");
  expect((await read(page)).dependencies).toHaveLength(2);
  const before = await read(page);
  const a = await page.locator('[data-task="a"] .right').boundingBox();
  await page.mouse.move(a.x + 4, a.y + 20);
  await page.mouse.down();
  await page.mouse.move(a.x + 116, a.y + 20, { steps: 10 });
  await page.mouse.up();
  let s = await read(page);
  expect(s.tasks[1].start).toBe(before.tasks[1].start + 7 * DAY);
  expect(s.tasks[2].end).toBe(before.tasks[2].end + 7 * DAY);
  await link(page, "c", "a");
  expect((await read(page)).dependencies).toHaveLength(2);
  await expect(page.locator("#notice")).toContainText("循环");
  const r = await page.locator('[data-task="a"]').boundingBox();
  await page.mouse.move(r.x + 30, r.y + 20);
  await page.mouse.down();
  await page.mouse.move(r.x + 142, r.y + 20, { steps: 8 });
  await page.keyboard.press("Escape");
  await page.mouse.up();
  expect(await read(page)).toEqual(s);
  await page.locator('[data-task="b"]').dblclick();
  await page.getByRole("button", { name: "解除", exact: true }).first().click();
  expect((await read(page)).dependencies).toHaveLength(1);
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page.locator(".dependency-line")).toHaveCount(1);
});
test("关键点创建编辑删除及删除空行", async ({ page }) => {
  await seed(page);
  await page.locator("#addMilestone").click();
  await page.getByLabel("关键点名称").fill("发布评审");
  await page.getByLabel("重要时间").fill("2026-11-05T10:30");
  await page.getByRole("button", { name: "保存更改" }).click();
  await expect(page.locator(".milestone")).toHaveText(/发布评审/);
  await expect(page.locator(".milestone-line")).toHaveCount(1);
  expect((await read(page)).milestones[0].time).toBe(
    Date.UTC(2026, 10, 5, 10, 30),
  );
  await page.locator(".milestone").click();
  await page.getByLabel("关键点名称").fill("正式发布");
  await page.getByRole("button", { name: "保存更改" }).click();
  await page.getByRole("button", { name: "项目：删除第 4 空行" }).click();
  expect((await read(page)).categories[0].rows).toBe(3);
  await expect(page.getByRole("button", { name: /删除第/ })).toHaveCount(0);
  await page.locator(".milestone").click();
  await page.getByRole("button", { name: "删除关键点", exact: true }).click();
  await expect(page.locator(".milestone")).toHaveCount(0);
});
test("PNG、SVG 与 JSON 导出包含完整数据且可解码", async ({
  page,
}, testInfo) => {
  const s = data();
  s.dependencies = [{ from: "a", to: "b" }];
  s.milestones = [{ id: "m", title: "发布节点", time: base + 14 * DAY }];
  await seed(page, s);
  for (const format of ["svg", "png", "json"]) {
    await page.locator("#export").click();
    const pending = page.waitForEvent("download");
    await page
      .getByRole("menuitem", { name: new RegExp(format.toUpperCase()) })
      .click();
    const d = await pending;
    const path = testInfo.outputPath("timeline." + format);
    await d.saveAs(path);
    const bytes = await readFile(path);
    if (format === "svg") {
      const svg = bytes.toString();
      expect(svg).toContain("发布节点");
      expect(svg).toContain('stroke-dasharray="5 5"');
      expect(svg).toContain("测试");
      const valid = await page.evaluate(
        (svg) =>
          new DOMParser()
            .parseFromString(svg, "image/svg+xml")
            .querySelector("parsererror") === null,
        svg,
      );
      expect(valid).toBe(true);
    } else if (format === "png") {
      expect(bytes.subarray(1, 4).toString()).toBe("PNG");
      expect(bytes.readUInt32BE(16)).toBeGreaterThan(1500);
      expect(bytes.readUInt32BE(20)).toBeGreaterThan(900);
    } else {
      const parsed = JSON.parse(bytes);
      expect(parsed.dependencies).toEqual(s.dependencies);
      expect(parsed.milestones).toEqual(s.milestones);
    }
  }
});

test("关键点拖动吸附、Ctrl 精调和取消，行删除直接在图中", async ({ page }) => {
  const s = data();
  s.milestones = [{ id: "m", title: "评审", time: base }];
  await seed(page, s);
  const move = async (dx, modifier, cancel = false) => {
    const r = await page.locator(".milestone").boundingBox();
    if (modifier) await page.keyboard.down(modifier);
    await page.mouse.move(r.x + 20, r.y + 12);
    await page.mouse.down();
    await page.mouse.move(r.x + 20 + dx, r.y + 12, { steps: 8 });
    if (cancel) await page.keyboard.press("Escape");
    await page.mouse.up();
    if (modifier) await page.keyboard.up(modifier);
  };
  await move(112);
  expect((await read(page)).milestones[0].time).toBe(base + 7 * DAY);
  await expect(page.locator("dialog")).not.toBeVisible();
  await move(25, "Control");
  const time = (await read(page)).milestones[0].time;
  expect(time).toBeGreaterThan(base + 7 * DAY);
  expect((time - base) % (7 * DAY)).not.toBe(0);
  await move(112, null, true);
  expect((await read(page)).milestones[0].time).toBe(time);
  await page.locator(".edit-category").click();
  await expect(page.locator("dialog [data-delete-row]")).toHaveCount(0);
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await page.locator("#canvas [data-delete-row]").click();
  expect((await read(page)).categories[0].rows).toBe(3);
  await expect(page.locator("dialog")).not.toBeVisible();
});
