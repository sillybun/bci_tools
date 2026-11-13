import {test,expect} from '@playwright/test';
test('依赖矛盾导出提示可取消或继续，四种格式均检查',async({page})=>{
 const base=Date.UTC(2026,10,2),week=7*86400000;
 await page.addInitScript(({base,week})=>localStorage.setItem('gantt-studio-v1',JSON.stringify({categories:[{id:'x',name:'类别',rows:2}],tasks:[{id:'a',title:'前项',categoryId:'x',row:0,start:base,end:base+2*week},{id:'b',title:'后项',categoryId:'x',row:1,start:base+week,end:base+3*week}],dependencies:[{from:'a',to:'b'}]})),{base,week});
 await page.goto('/');
 const downloads=[];page.on('download',d=>downloads.push(d));
 for(const format of ['svg','png','jpg','json']){
  await page.locator('#export').click();
  await page.locator('[data-export="'+format+'"]').click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.locator('.conflict-notes')).toContainText('前项');
  await expect(page.getByRole('button',{name:'取消导出'})).toBeFocused();
  if(format==='png') await page.keyboard.press('Escape');
  else await page.getByRole('button',{name:'取消导出'}).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.locator('#export')).toBeFocused();
  await expect(page.locator('#export')).toBeEnabled();
 }
 expect(downloads).toHaveLength(0);
 await page.locator('#export').click();
 const download=page.waitForEvent('download');
 await page.locator('[data-export="svg"]').click();
 await page.getByRole('button',{name:'继续导出'}).click();
 expect((await download).suggestedFilename()).toMatch(/svg$/);
});
for(const compact of [false,true])test('类别排序及保存撤销 compact='+compact,async({page})=>{
 await page.goto('/');if(compact)await page.locator('#compactMode').check();
 const read=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('gantt-studio-v1')));
 const before=await read();
 const a=await page.locator('[data-reorder="design"]').boundingBox(),b=await page.locator('.category[data-category="launch"]').boundingBox();
 await page.mouse.move(a.x+8,a.y+12);await page.mouse.down();await page.mouse.move(a.x+8,b.y+b.height-4,{steps:10});await page.mouse.up();
 expect((await read()).categories.map(c=>c.id)).toEqual(['build','launch','design']);
 expect((await read()).tasks).toEqual(before.tasks);
 await page.locator('#undo').click();expect((await read()).categories.map(c=>c.id)).toEqual(['design','build','launch']);
 await page.locator('#redo').click();await page.reload();
 expect(await page.locator('.category').evaluateAll(els=>els.map(e=>e.dataset.category))).toEqual(['build','launch','design']);
 const grip=await page.locator('[data-reorder="design"]').boundingBox(),top=await page.locator('.category').first().boundingBox();
 await page.mouse.move(grip.x+8,grip.y+12);await page.mouse.down();await page.mouse.move(grip.x+8,top.y+2,{steps:10});await page.keyboard.press('Escape');await page.mouse.up();
 expect((await read()).categories.map(c=>c.id)).toEqual(['build','launch','design']);
 await expect(page.locator('#fontSize')).toHaveCSS('padding-left','12px');
 await expect(page.locator('#fontSize')).toHaveCSS('padding-right','24px');
});
