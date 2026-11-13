import test from 'node:test';
import assert from 'node:assert/strict';
import {validate,imageState,categoryColor,WEEK} from '../src/model.js';
import {projectSVG} from '../src/export.js';
const start=Date.UTC(2026,10,2);
const sample=()=>({categories:[{id:'a',name:'Hidden category',rows:2,exportVisible:false,color:'#aa4400'},{id:'b',name:'Visible category',rows:1}],tasks:[{id:'a',title:'Hidden task',categoryId:'a',row:0,start:start-20*WEEK,end:start-19*WEEK},{id:'b',title:'Visible task',categoryId:'b',row:0,start,end:start+WEEK}],dependencies:[{from:'a',to:'b'}]});
test('图片过滤不修改原项目，保留原类别配色并去掉关联依赖、时间范围与空行',()=>{
 const s=sample(), before=structuredClone(s), view=imageState(s);
 assert.deepEqual(s,before);assert.equal(view.tasks.length,1);assert.equal(view.dependencies.length,0);
 assert.equal(view.categories[0].color,categoryColor(s.categories[1],1));
 const result=projectSVG(s);assert.ok(!result.svg.includes('Hidden'));assert.ok(result.svg.includes('Visible task'));assert.ok(result.svg.includes('#299c91'));assert.ok(result.width<1000);
 s.categories[1].exportVisible=false;
 assert.ok(projectSVG(s).svg.includes('0 个任务'));
});
test('旧格式兼容，自定义色和可见性校验与导出',()=>{
 const s=sample();s.categories[0].exportVisible=true;
 assert.equal(validate(s).categories[0].color,'#aa4400');assert.ok(projectSVG(s).svg.includes('stroke="#aa4400"'));
 s.categories[0].color='red;';assert.throws(()=>validate(s),/类别/);
 s.categories[0].color='#abcdef';s.categories[0].exportVisible='false';assert.throws(()=>validate(s),/类别/);
});
