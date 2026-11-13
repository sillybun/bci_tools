import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
test("标题改名、多项目隔离与自动恢复，每个项目独立显示设置", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#projectTitle").dblclick();
  await page.getByLabel("项目名称", { exact: true }).fill("春季计划");
  await page.keyboard.press("Enter");
  await page.getByLabel("显示任务时间区间", { exact: true }).uncheck();
  await page.locator("#fontSize").selectOption("18");
  await expect(page.locator(".task small")).toHaveCount(0);
  await expect(page.locator(".task strong").first()).toHaveCSS(
    "font-size",
    "18px",
  );
  await page.locator("#newProject").click();
  await expect(page.locator(".task")).toHaveCount(0);
  await expect(page.locator("#showTaskDates")).toBeChecked();
  await expect(page.locator("#fontSize")).toHaveValue("14");
  await page.locator("#projectTitle").dblclick();
  await page.getByLabel("项目名称", { exact: true }).fill("夏季计划");
  await page.keyboard.press("Enter");
  await page.reload();
  await expect(page.locator("#projectTitle")).toHaveText("夏季计划");
  await page.getByRole("button", { name: "项目列表", exact: true }).click();
  await page.getByRole("button", { name: /春季计划.*5 个任务/ }).click();
  await expect(page.locator("#projectTitle")).toHaveText("春季计划");
  await expect(page.locator(".task")).toHaveCount(5);
  await expect(page.locator("#showTaskDates")).not.toBeChecked();
  await expect(page.locator("#fontSize")).toHaveValue("18");
  await page.reload();
  await expect(page.locator("#projectTitle")).toHaveText("春季计划");
});
test("显示设置同步到 SVG/PNG/JPG，JPG 可解码且扩展名正确", async ({
  page,
}, info) => {
  await page.goto("/");
  await page.locator("#showTaskDates").uncheck();
  await page.locator("#fontSize").selectOption("18");
  for (const format of ["SVG", "PNG", "JPG"]) {
    await page.locator("#export").click();
    const pending = page.waitForEvent("download");
    await page
      .getByRole("menuitem", { name: new RegExp("^" + format) })
      .click();
    const download = await pending,
      path = info.outputPath("project." + format.toLowerCase());
    await download.saveAs(path);
    const bytes = await readFile(path);
    if (format === "SVG") {
      const content = bytes.toString();
      expect(content).toContain('font-size="15.42857142857143"');
      expect(content).not.toMatch(/\d{2}-\d{2} — \d{2}-\d{2}/);
    } else {
      const url =
        "data:image/" +
        (format === "JPG" ? "jpeg" : "png") +
        ";base64," +
        bytes.toString("base64");
      const size = await page.evaluate(async (url) => {
        const img = new Image();
        img.src = url;
        await img.decode();
        return [img.width, img.height];
      }, url);
      expect(size[0]).toBeGreaterThan(1000);
      if (format === "JPG") {
        expect(download.suggestedFilename()).toMatch(/\.jpg$/);
        expect([...bytes.subarray(0, 3)]).toEqual([255, 216, 255]);
      }
    }
  }
});
