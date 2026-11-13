import { test, expect } from "@playwright/test";
test("撤销重做快捷键、设置恢复、项目历史隔离和输入框原生行为", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#undo")).toBeDisabled();
  await page.locator("#showTaskDates").uncheck();
  await page.keyboard.press("Control+z");
  await expect(page.locator("#showTaskDates")).toBeChecked();
  await page.keyboard.press("Control+Shift+z");
  await expect(page.locator("#showTaskDates")).not.toBeChecked();
  await page.locator("#fontSize").selectOption("18");
  await page.locator("#undo").click();
  await expect(page.locator("#fontSize")).toHaveValue("14");
  await page.locator("#redo").click();
  await expect(page.locator("#fontSize")).toHaveValue("18");
  await page.locator("#newProject").click();
  await expect(page.locator("#undo")).toBeDisabled();
  await page.locator("#projectsButton").click();
  await page.getByRole("button", { name: /项目时间线.*5 个任务/ }).click();
  await page.locator("#undo").click();
  await expect(page.locator("#fontSize")).toHaveValue("14");
  await page.locator("#projectTitle").dblclick();
  await page.locator("#projectNameInput").fill("输入中");
  await page.keyboard.press("Control+z");
  await expect(page.locator("#showTaskDates")).not.toBeChecked();
  await page.keyboard.press("Escape");
});
test("一次任务拖动可整体撤销，连线颜色与细箭头同步导出", async ({ page }) => {
  await page.goto("/");
  const before = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("gantt-studio-v1")),
  );
  const r = await page.locator('[data-task="c"]').boundingBox();
  await page.mouse.move(r.x + 30, r.y + 20);
  await page.mouse.down();
  await page.mouse.move(r.x + 142, r.y + 20, { steps: 12 });
  await page.mouse.up();
  await page.locator("#undo").click();
  expect(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("gantt-studio-v1")),
    ),
  ).toEqual(before);
  await expect(page.locator("#undo")).toBeDisabled();
  const a = await page.locator('[data-task="a"]').boundingBox(),
    b = await page.locator('[data-task="c"]').boundingBox();
  await page.keyboard.down("Shift");
  await page.mouse.move(a.x + 30, a.y + 20);
  await page.mouse.down();
  await page.mouse.move(b.x + 30, b.y + 20, { steps: 8 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await expect(page.locator(".dependency-line")).toHaveCSS(
    "stroke",
    "rgb(105, 97, 217)",
  );
  await expect(page.locator("#depArrow0")).toHaveAttribute(
    "markerUnits",
    "userSpaceOnUse",
  );
  await expect(page.locator("#depArrow0 path")).toHaveAttribute("fill", "none");
  await page.locator("#undo").click();
  await expect(page.locator(".dependency-line")).toHaveCount(0);
  await page.locator("#redo").click();
  await expect(page.locator(".dependency-line")).toHaveCount(1);
});
