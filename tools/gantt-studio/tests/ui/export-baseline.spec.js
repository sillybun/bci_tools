import {test,expect} from '@playwright/test';

test('导出标题采用显式基线，在普通与紧凑模式及所有字号下保持居中',async({page})=>{
 await page.goto('/');
 for(const compact of [false,true]) for(const fontSize of [12,14,16,18]){
  const result=await page.evaluate(async({compact,fontSize})=>{
   const {projectSVG}=await import('/src/export.js');
   const state={name:'基线验证',categories:[{id:'c',name:'类别',rows:1}],tasks:[{id:'t',categoryId:'c',row:0,title:'轮椅控制器 ABC',start:Date.UTC(2026,10,2),end:Date.UTC(2026,10,16)}],settings:{compact,fontSize,showTaskDates:false}};
   const host=document.createElement('div');
   host.innerHTML=projectSVG(state).svg;
   document.body.append(host);
   await document.fonts.ready;
   const label=host.querySelector('g[data-task-text] text');
   const rect=host.querySelector('clipPath rect');
   const box=label.getBBox();
   const top=Number(rect.getAttribute('y')),height=Number(rect.getAttribute('height'));
   const result={offset:Math.abs(box.y+box.height/2-top-height/2),inside:box.y>=top&&box.y+box.height<=top+height,baseline:label.hasAttribute('dominant-baseline')};
   host.remove();
   return result;
  },{compact,fontSize});
  expect(result.baseline).toBe(false);
  expect(result.offset).toBeLessThan(2);
  expect(result.inside).toBe(true);
 }
});
