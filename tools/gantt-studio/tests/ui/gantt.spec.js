import { test, expect } from "@playwright/test";
const DAY = 86400000,
  WEEK = 7 * DAY;
async function read(page) {
  return page.evaluate(() =>
    JSON.parse(localStorage.getItem("gantt-studio-v1")),
  );
}
async function drag(page, loc, dx, dy = 0, modifier) {
  const r = await loc.boundingBox();
  if (modifier) await page.keyboard.down(modifier);
  await page.mouse.move(r.x + r.width / 2, r.y + r.height / 2);
  await page.mouse.down();
  await page.mouse.move(r.x + r.width / 2 + dx, r.y + r.height / 2 + dy, {
    steps: 12,
  });
  await page.mouse.up();
  if (modifier) await page.keyboard.up(modifier);
}
test("创建类别、任务编辑、持久化及导入导出", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "＋ 新建类别" }).click();
  await page.getByLabel("类别名称").fill("新类别");
  await page.getByLabel("类别说明").fill("测试说明");
  await page.getByRole("button", { name: "保存更改" }).click();
  await expect(page.locator(".category-label strong").last()).toHaveText(
    "新类别",
  );
  await page.locator('[data-task="a"]').dblclick();
  await page.getByLabel("任务名称").fill("新的任务名");
  await page.getByLabel("任务内容").fill("任务的详细内容");
  await page.getByRole("button", { name: "保存更改" }).click();
  await page.reload();
  await expect(page.locator('[data-task="a"] strong')).toHaveText("新的任务名");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "↧ 导出项目 ▾" }).click();
  await page.getByRole("menuitem", { name: /JSON/ }).click();
  expect((await download).suggestedFilename()).toMatch(/项目时间线/);
  page.on("dialog", (d) => d.accept());
  await page.locator("#file").setInputFiles({
    name: "project.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({
        categories: [{ id: "x", name: "导入类别", rows: 1 }],
        tasks: [],
      }),
    ),
  });
  await expect(page.locator(".category-label strong")).toHaveText(["导入类别"]);
});
test("双击空白新建任务", async ({ page }) => {
  await page.goto("/");
  const track = page.locator(".track").first();
  const r = await track.boundingBox();
  const view = await page.locator("#viewport").boundingBox();
  await page.mouse.dblclick(view.x + 650, r.y + 45);
  await expect(page.locator("dialog")).toBeVisible();
  await page.getByLabel("任务名称").fill("双击创建");
  await page.getByRole("button", { name: "保存更改" }).click();
  await expect(
    page.locator(".task strong", { hasText: "双击创建" }),
  ).toBeVisible();
  expect((await read(page)).tasks).toHaveLength(6);
});
test("周吸附、自由调整、重叠分行及边沿插行", async ({ page }) => {
  await page.goto("/");
  await drag(page, page.locator('[data-task="c"]'), 120);
  let s = await read(page);
  let c = s.tasks.find((t) => t.id === "c");
  expect(new Date(c.start).getUTCDay()).toBe(1);
  const oldEnd = c.end;
  await drag(page, page.locator('[data-task="c"] .right'), 25, 0, "Control");
  s = await read(page);
  c = s.tasks.find((t) => t.id === "c");
  expect(c.end).toBeGreaterThan(oldEnd);
  expect((c.end - oldEnd) % WEEK).not.toBe(0);
  await drag(page, page.locator('[data-task="b"]'), 0, -64);
  s = await read(page);
  expect(s.categories[0].rows).toBe(3);
  const r = await page.locator('[data-task="b"]').boundingBox(),
    track = await page.locator(".track").first().boundingBox();
  await page.mouse.move(r.x + r.width / 2, r.y + 20);
  await page.mouse.down();
  await page.mouse.move(r.x + r.width / 2, track.y + 5, { steps: 10 });
  await page.mouse.up();
  s = await read(page);
  expect(s.categories[0].rows).toBe(4);
  expect(s.tasks.find((t) => t.id === "b").row).toBe(0);
});
test("缩放、平移且无运行时错误", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "放大", exact: true }).click();
  await expect(page.locator("#zoomLabel")).toHaveText("125%");
  const before = await page.locator("#viewport").evaluate((e) => e.scrollLeft);
  const rect = await page.locator(".ticks").boundingBox(),
    view = await page.locator("#viewport").boundingBox();
  await page.mouse.move(view.x + 700, rect.y + 30);
  await page.mouse.down();
  await page.mouse.move(view.x + 600, rect.y + 30, { steps: 8 });
  await page.mouse.up();
  expect(
    await page.locator("#viewport").evaluate((e) => e.scrollLeft),
  ).toBeGreaterThan(before);
  expect(errors).toEqual([]);
  await page.screenshot({
    path: "test-results/gantt-studio.png",
    fullPage: true,
  });
});

test("左边框缩放、Command 自由调整、取消拖动与跨类别移动", async ({ page }) => {
  await page.goto("/");
  const initial = await page.locator('[data-task="c"]').boundingBox();
  await drag(page, page.locator('[data-task="c"] .left'), 35, 0, "Meta");
  let s = await read(page);
  let c = s.tasks.find((t) => t.id === "c");
  expect(new Date(c.start).getUTCHours()).not.toBe(0);
  const saved = c.start;
  const r = await page.locator('[data-task="c"]').boundingBox();
  await page.mouse.move(r.x + 30, r.y + 20);
  await page.mouse.down();
  await page.mouse.move(r.x + 150, r.y + 20, { steps: 5 });
  await page.keyboard.press("Escape");
  await page.mouse.up();
  expect((await read(page)).tasks.find((t) => t.id === "c").start).toBe(saved);
  const task = await page.locator('[data-task="c"]').boundingBox(),
    target = await page.locator(".track").first().boundingBox();
  await page.mouse.move(task.x + 30, task.y + 20);
  await page.mouse.down();
  await page.mouse.move(task.x + 30, target.y + 40, { steps: 10 });
  await page.mouse.up();
  s = await read(page);
  c = s.tasks.find((t) => t.id === "c");
  expect(c.categoryId).toBe("design");
  await expect(page.locator("#file")).toBeHidden();
});
