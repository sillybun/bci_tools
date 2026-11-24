import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const src=JSON.parse(await fs.readFile(process.argv[2],'utf8')), output=process.argv[3];
const wb=Workbook.create();let counter=0;
function col(n){let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;}
function value(v){if(v===undefined || v===null)return 'n.a.';return typeof v==='object'?JSON.stringify(v):v;}
function table(name,rows,width=20){
 const sh=wb.worksheets.add(name),end=col(rows[0].length),n=rows.length;
 sh.showGridLines=false;const range=sh.getRange(`A1:${end}${n}`);range.values=rows.map(r=>r.map(value));
 range.format.font={name:'Microsoft YaHei',size:11,color:'#18232E'};range.format.rowHeight=25;range.format.columnWidth=width;
 const header=sh.getRange(`A1:${end}1`);header.format.fill='#24496B';header.format.font={bold:true,color:'#FFFFFF'};header.format.rowHeight=52;header.format.wrapText=true;header.format.horizontalAlignment='center';
 sh.freezePanes.freezeRows(1);sh.freezePanes.freezeColumns(3);
 if(n>1)sh.tables.add(`A1:${end}${n}`,true,'Comparison'+counter++);
 return sh;
}
const labels={plot_id:'图编号',date:'日期',task:'任务',run:'Run',control:'控制方式',status:'数据状态',trials:'Trial 数',successes:'成功数',success_rate:'成功率',windows:'BPS 窗口数',bps_median:'Grid BPS 中位数 (bits/s)',bps_q1:'Grid BPS Q1',bps_q3:'Grid BPS Q3',bps_peak:'Grid BPS 峰值',rt_median_s:'TTT 中位数 (s)',rt_q1_s:'TTT Q1 (s)',rt_q3_s:'TTT Q3 (s)',grid:'Grid 行×列',dwell_s:'Dwell (s)',timeout_s:'超时 (s)',preview_next_target:'是否显示 next target',nearby_assist:'邻近目标辅助',joystick_gain:'摇杆位移增益',bci_velocity_scale:'脑控速度倍率',bci_smooth_alpha:'脑控平滑 alpha',bci_cursor_mode:'脑控光标模式',configured_model:'配置模型',online_update:'总在线更新开关',active_model_update:'当前模型更新开关',blocks_per_update:'更新间隔 blocks',min_blocks_for_refit:'最少 refit blocks',blend_alpha:'更新混合 alpha',auto_swap:'自动替换模型',blocks:'计划 block 数',config_source:'Config 来源',run_folder:'Run 原始目录',config_key:'配置对应键',excel_row:'Excel 行号',rt_n:'TTT 有效成功数'};
const preferred=['plot_id','date','task','run','control','status','trials','successes','success_rate','windows','bps_median','bps_q1','bps_q3','bps_peak','rt_median_s','rt_q1_s','rt_q3_s','grid','dwell_s','timeout_s','preview_next_target','nearby_assist','joystick_gain','bci_velocity_scale','bci_smooth_alpha','bci_cursor_mode','configured_model','online_update','active_model_update','blocks_per_update','min_blocks_for_refit','blend_alpha','auto_swap','blocks'];
const keys=[...preferred,...new Set(src.rows.flatMap(Object.keys).filter(k=>!preferred.includes(k)))];
const rows=[keys.map(k=>labels[k]||k),...src.rows.map(r=>keys.map(k=>value(r[k])))];
const sh=table('逐run统计',rows);
for(let j=0;j<keys.length;j++){
 const k=keys[j],c=col(j+1),range=sh.getRange(`${c}2:${c}${rows.length}`);
 if(k==='date'){range.values=src.rows.map(r=>[(Date.parse(r.date+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000]);range.setNumberFormat('yyyy-mm-dd');}
 else if(k==='run')range.setNumberFormat('000');
 else if(k==='plot_id')range.setNumberFormat('00');
 else if(k==='success_rate')range.setNumberFormat('0.0%');
 else if(/median|q1|q3|peak|dwell|timeout|gain|alpha|scale/.test(k))range.setNumberFormat('0.000');
 if(/source|folder|config_key/.test(k))sh.getRange(`${c}1:${c}${rows.length}`).format.columnWidth=65;
}
for(let i=0;i<src.rows.length;i++){
 const r=src.rows[i],c=r.control==='BCI'?'#0072B2':r.control==='Joystick'?'#D55E00':'#777777';
 sh.getRange(`A${i+2}:F${i+2}`).format.font={color:c,bold:true};
}
const latest=src.rows.map(r=>r.date).sort().at(-1);
const changes=src.changes.filter(r=>r.config_key.startsWith(latest));
const ch=table('当天配置变化',[['Run 配置对应键','相对前一个 run','配置字段','之前','当前'],...changes.map(r=>[r.config_key,r.previous,r.field,r.before,r.after])],34);
ch.getRange(`C1:C${changes.length+1}`).format.columnWidth=68;
ch.getRange(`A2:E${changes.length+1}`).format.wrapText=true;
ch.getRange(`A2:E${changes.length+1}`).format.rowHeight=50;
const fields=[...new Set(src.full_config.flatMap(r=>Object.keys(r.fields)))].sort();
const all=table('完整配置对比',[['配置字段','中文说明','适用范围',...src.full_config.map(r=>r.config_key)],...fields.map(k=>[k,k==='GRID_TASK.PREVIEW_NEXT_TARGET'?'是否显示下一个目标':k==='CONTROL.POS_SCALE_FACTOR'?'摇杆位移增益':k==='CONTROL.BCI_VELOCITY_SCALE'?'脑控速度倍率':'',k.startsWith('GRID_TASK.')?'Grid':k.startsWith('CENTER_OUT_TASK.')?'CO8':'共用 / 配置保存值',...src.full_config.map(r=>Object.hasOwn(r.fields,k)?(r.fields[k]===null?'null（保存值）':value(r.fields[k])):'未保存字段')])],25);
all.getRange(`A1:A${fields.length+1}`).format.columnWidth=72;
all.getRange(`D1:${col(src.full_config.length+3)}1`).format.rowHeight=75;
const notes=table('方法与来源',[
 ['项目','说明'],
 ['范围',`${src.rows[0].date} 至 ${latest}；排除 1116；仅正式编号 run，CSV 独立于 XDF 是否可用。`],
 ['对应关系','图的 #编号 = 逐run统计的图编号；配置对应键 = 日期|任务|run；完整配置表保留所有相同字段。'],
 ['颜色','蓝色圆点：脑控 BCI；橙色方点：离线摇杆 Joystick；灰色：配置或行为数据缺失。'],
 ['BPS',src.method],
 ['TTT','每个成功 trial 仅统计一次，包含 dwell；中位数、Q1、Q3；误差线是 IQR，不是置信区间。'],
 ['CO8','Neuralink / Wolpaw / Fitts 分开显示，不与 Grid BPS 混为同一口径。'],
 ['缺失值','n.a. 为不可用或不适用；未保存字段为旧 config 无此字段；null（保存值）为原始空值。'],
 ['模型解释','配置模型和在线更新是保存设置，不能证明外部解码器实际更新成功。'],
 ['速度解释','POS_SCALE_FACTOR 仅为摇杆位移增益；BCI_VELOCITY_SCALE 为脑控速度倍率；两者不是同一个参数。'],
 ['重复会话','1111 Grid run003-b 与 run004 元信息指向同一 session；003-b 缺独立 CSV，不复制其他 run 的指标。'],
 ...src.full_config.map(r=>[r.config_key,r.source])
 ],75);
notes.getRange(`A1:A${src.full_config.length+12}`).format.columnWidth=30;
notes.getRange(`B1:B${src.full_config.length+12}`).format.columnWidth=110;
notes.getRange(`A2:B${src.full_config.length+12}`).format.wrapText=true;
notes.getRange(`A2:B${src.full_config.length+12}`).format.rowHeight=48;
notes.getRange('A5:B5').format.rowHeight=180;
await wb.recalculate();
console.log(JSON.stringify(await wb.inspect({kind:'table',range:"'逐run统计'!A1:J5",include:'values',tableMaxRows:5,tableMaxCols:10})));
const preview=await wb.render({sheetName:'逐run统计',range:'A1:J12',scale:1.3,format:'png'});
await fs.writeFile(output.replace(/\.xlsx$/, '_preview.png'),new Uint8Array(await preview.arrayBuffer()));
for(const [sheetName,range,suffix] of [['当天配置变化','A1:E10','changes'],['完整配置对比','A1:F10','config'],['方法与来源','A1:B7','methods']]){
 const image=await wb.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(output.replace(/\.xlsx$/, '_'+suffix+'_preview.png'),new Uint8Array(await image.arrayBuffer()));
}
const book=await SpreadsheetFile.exportXlsx(wb);await book.save(output);
