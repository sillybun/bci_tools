import {test,expect} from '@playwright/test';
test('统计折叠、紧凑行高、拖动分行与独立保存',async({page})=>{
 await page.goto('/');const height=await page.locator('.category').first().evaluate(e=>e.getBoundingClientRect().height);
 await page.locator('#toggleStats').click();await expect(page.locator('#projectStats')).toBeHidden();await expect(page.locator('#toggleStats')).toHaveAttribute('aria-expanded','false');
 await page.locator('#compactMode').check();await expect(page.locator('#showTaskDates')).not.toBeChecked();expect(await page.locator('.category').first().evaluate(e=>e.getBoundingClientRect().height)).toBe(height/2);await expect(page.locator('.task').first()).toHaveCSS('height','24px');
 const r=await page.locator('[data-task="b"]').boundingBox();await page.mouse.move(r.x+30,r.y+12);await page.mouse.down();await page.mouse.move(r.x+30,r.y+12-32,{steps:8});await page.mouse.up();
 const state=await page.evaluate(()=>JSON.parse(localStorage.getItem('gantt-studio-v1')));expect(state.categories[0].rows).toBe(3);
 await page.reload();await expect(page.locator('#compactMode')).toBeChecked();await expect(page.locator('#projectStats')).toBeHidden();await page.locator('#compactMode').uncheck();await expect(page.locator('.task').first()).toHaveCSS('height','48px');await page.locator('#toggleStats').click();await expect(page.locator('#projectStats')).toBeVisible();
});

test('紧凑页眉移动、恢复与导出隔离', async ({page}) => {
 await page.goto('/');
 await expect(page.locator('header .crumb')).toBeVisible();
 await page.locator('#compactMode').check();
 await expect(page.locator('main > header')).toBeHidden();
 await expect(page.locator('.intro .crumb')).toBeVisible();
 await expect(page.locator('.actions > #saved')).toBeVisible();
 const positions = await page.evaluate(() => {
   const box = s => document.querySelector(s).getBoundingClientRect().toJSON();
   return {crumb:box('.crumb'), title:box('h1'), saved:box('#saved'), button:box('#newProject')};
 });
 expect(positions.crumb.right).toBeLessThanOrEqual(positions.title.x);
 expect(positions.saved.right).toBeLessThanOrEqual(positions.button.x);
 const svg = await page.evaluate(async () => {
   const {projectSVG} = await import('/src/export.js');
   return projectSVG(JSON.parse(localStorage.getItem('gantt-studio-v1'))).svg;
 });
 expect(svg).toContain('GANTT STUDIO');
 expect(svg).not.toContain('所有更改已保存');
 expect(svg).not.toContain('category-grip');
 await page.reload();
 await expect(page.locator('.actions > #saved')).toBeVisible();
 await page.locator('#compactMode').uncheck();
 await expect(page.locator('header .crumb')).toBeVisible();
 await expect(page.locator('header #saved')).toBeVisible();
});


for (const compact of [false,true]) test('类别拖拽排序、撤销重做与保存 compact='+compact, async ({page}) => {
 await page.goto('/');
 if(compact) await page.locator('#compactMode').check();
 const read = () => page.evaluate(() => JSON.parse(localStorage.getItem('gantt-studio-v1')));
 const before = await read();
 const handle = await page.locator('[data-reorder="design"]').boundingBox();
 const target = await page.locator('.category[data-category="launch"]').boundingBox();
 await page.mouse.move(handle.x+8,handle.y+12);
 await page.mouse.down();
 await page.mouse.move(handle.x+8,target.y+target.height-4,{steps:10});
 await page.mouse.up();
 const after = await read();
 expect(after.categories.map(c=>c.id)).toEqual(['build','launch','design']);
 expect(after.tasks).toEqual(before.tasks);
 expect(after.dependencies).toEqual(before.dependencies);
 await page.locator('#undo').click();
 expect((await read()).categories.map(c=>c.id)).toEqual(['design','build','launch']);
 await page.locator('#redo').click();
 await page.reload();
 expect(await page.locator('.category').evaluateAll(els=>els.map(e=>e.dataset.category))).toEqual(['build','launch','design']);
 const grip = await page.locator('[data-reorder="design"]').boundingBox();
 const first = await page.locator('.category').first().boundingBox();
 await page.mouse.move(grip.x+8,grip.y+12);
 await page.mouse.down();
 await page.mouse.move(grip.x+8,first.y+2,{steps:10});
 await page.keyboard.press('Escape');
 await page.mouse.up();
 expect((await read()).categories.map(c=>c.id)).toEqual(['build','launch','design']);
 await page.locator('[data-reorder="design"]').focus();
 await page.keyboard.press('ArrowUp');
 expect((await read()).categories.map(c=>c.id)).toEqual(['build','design','launch']);
});
