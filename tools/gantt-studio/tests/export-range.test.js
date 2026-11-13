import test from 'node:test';
import assert from 'node:assert/strict';
import {projectSVG} from '../src/export.js';
import {DAY,WEEK} from '../src/model.js';
const base=Date.UTC(2026,10,2);
function sample(end,milestones=[]){
 return {categories:[{id:'c',name:'类别',rows:1}],tasks:[{id:'t',title:'任务',categoryId:'c',row:0,start:base,end}],milestones};
}
test('周一结束的短任务左右各留一列，不再强制六列',()=>{
 const {svg,width}=projectSVG(sample(base+WEEK));
 assert.equal(width,264+3*150+40);
 assert.match(svg,/x="414" y="232" width="150"/);
});
test('周中结束先取整至周边界，再留一列',()=>{
 assert.equal(projectSVG(sample(base+WEEK+DAY)).width,264+4*150+40);
});
test('更早和更晚的关键点也纳入留白边界',()=>{
 const {width}=projectSVG(sample(base+WEEK,[{title:'早',time:base-WEEK},{title:'晚',time:base+2*WEEK}]));
 assert.equal(width,264+5*150+40);
});
