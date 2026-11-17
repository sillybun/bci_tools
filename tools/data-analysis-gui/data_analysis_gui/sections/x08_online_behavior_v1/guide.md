# X08 · 在线行为表现对比

**一句话总结：**CO8与Grid分开；成功率和到靶时间保持原口径，所有BPS按完整60秒有效时间窗口计算，再按run汇总median/IQR和peak。

## 图 1：Online CO8 Run Success and Time to Target

### 计算公式

**符号**

- W为累计60秒有效尝试时间窗口，T=60秒，C为窗口内完成的正确选择数，N为已核实的目标数。排除ITI、block休息和反馈等待，保留选择dwell、失败及超时耗时。

**步骤**

1. 按累计有效时间每1秒更新；长run从第60秒开始，整个run不足60秒时用一次实际全程窗口，ratio=T/60，BPS=bits/(60×ratio)。窗口左边界切入trial时只累计重叠时间，选择结局按完成时刻落在窗口内计数。不跨session、录制片段、缺失计时或trial编号断档建窗。
2. Grid仅画C×log₂(N)/60。CO8保留Neuralink式(C−明确错误选择)×log₂(7)/60和Wolpaw估算；Fitts使用同一60秒窗口内完整成功trial的难度总和/外向运动总时长，边界不完整trial不参与Fitts。
3. 每run的median/IQR和peak来自60秒、step 1秒窗口；仅整个run不足60秒时按实际时长折算，峰值表记录有效时间与trial范围。IQR表示窗口分布，不是置信区间。

### 结果解读

- **行为与BPS**：成功率与到靶时间继续按各图标明的block或run汇总。BPS曲线横轴为累计有效尝试分钟；run汇总的每个点对应一个run。Grid采用横向A4比例的上下两个子图，共用run横轴。上图为蓝色BPS median/IQR与橙色peak，下图为橘色成功trial到靶时间median/IQR（秒）。到靶时间按同run原始成功trial汇总，不平均滚动窗口的RT。绿色虚线为理想100%成功率下达到6 bits/s所需的平均TTT=log₂(N)/6，不是成功trial中位数的严格阈值；存在失败、超时耗时时需更快。

### 质量评价标准

- **行为与BPS**：整个run不足60秒时按实际时长折算；网格大小或Fitts输入无法核实时显示N/A，不把不完整开头当作峰值。超时不能删除。滚动窗口重叠，不是独立样本；Fitts与选择BPS的公式不同，不可直接互换。

## 图 2：Online CO8 Run Rolling BPS Median and IQR

### 计算公式

**符号**

- W为累计60秒有效尝试时间窗口，T=60秒，C为窗口内完成的正确选择数，N为已核实的目标数。排除ITI、block休息和反馈等待，保留选择dwell、失败及超时耗时。

**步骤**

1. 按累计有效时间每1秒更新；长run从第60秒开始，整个run不足60秒时用一次实际全程窗口，ratio=T/60，BPS=bits/(60×ratio)。窗口左边界切入trial时只累计重叠时间，选择结局按完成时刻落在窗口内计数。不跨session、录制片段、缺失计时或trial编号断档建窗。
2. Grid仅画C×log₂(N)/60。CO8保留Neuralink式(C−明确错误选择)×log₂(7)/60和Wolpaw估算；Fitts使用同一60秒窗口内完整成功trial的难度总和/外向运动总时长，边界不完整trial不参与Fitts。
3. 每run的median/IQR和peak来自60秒、step 1秒窗口；仅整个run不足60秒时按实际时长折算，峰值表记录有效时间与trial范围。IQR表示窗口分布，不是置信区间。

### 结果解读

- **行为与BPS**：成功率与到靶时间继续按各图标明的block或run汇总。BPS曲线横轴为累计有效尝试分钟；run汇总的每个点对应一个run。Grid采用横向A4比例的上下两个子图，共用run横轴。上图为蓝色BPS median/IQR与橙色peak，下图为橘色成功trial到靶时间median/IQR（秒）。到靶时间按同run原始成功trial汇总，不平均滚动窗口的RT。绿色虚线为理想100%成功率下达到6 bits/s所需的平均TTT=log₂(N)/6，不是成功trial中位数的严格阈值；存在失败、超时耗时时需更快。

### 质量评价标准

- **行为与BPS**：整个run不足60秒时按实际时长折算；网格大小或Fitts输入无法核实时显示N/A，不把不完整开头当作峰值。超时不能删除。滚动窗口重叠，不是独立样本；Fitts与选择BPS的公式不同，不可直接互换。

## 图 3：Online CO8 Run Rolling BPS Peak

### 计算公式

**符号**

- W为累计60秒有效尝试时间窗口，T=60秒，C为窗口内完成的正确选择数，N为已核实的目标数。排除ITI、block休息和反馈等待，保留选择dwell、失败及超时耗时。

**步骤**

1. 按累计有效时间每1秒更新；长run从第60秒开始，整个run不足60秒时用一次实际全程窗口，ratio=T/60，BPS=bits/(60×ratio)。窗口左边界切入trial时只累计重叠时间，选择结局按完成时刻落在窗口内计数。不跨session、录制片段、缺失计时或trial编号断档建窗。
2. Grid仅画C×log₂(N)/60。CO8保留Neuralink式(C−明确错误选择)×log₂(7)/60和Wolpaw估算；Fitts使用同一60秒窗口内完整成功trial的难度总和/外向运动总时长，边界不完整trial不参与Fitts。
3. 每run的median/IQR和peak来自60秒、step 1秒窗口；仅整个run不足60秒时按实际时长折算，峰值表记录有效时间与trial范围。IQR表示窗口分布，不是置信区间。

### 结果解读

- **行为与BPS**：成功率与到靶时间继续按各图标明的block或run汇总。BPS曲线横轴为累计有效尝试分钟；run汇总的每个点对应一个run。Grid采用横向A4比例的上下两个子图，共用run横轴。上图为蓝色BPS median/IQR与橙色peak，下图为橘色成功trial到靶时间median/IQR（秒）。到靶时间按同run原始成功trial汇总，不平均滚动窗口的RT。绿色虚线为理想100%成功率下达到6 bits/s所需的平均TTT=log₂(N)/6，不是成功trial中位数的严格阈值；存在失败、超时耗时时需更快。

### 质量评价标准

- **行为与BPS**：整个run不足60秒时按实际时长折算；网格大小或Fitts输入无法核实时显示N/A，不把不完整开头当作峰值。超时不能删除。滚动窗口重叠，不是独立样本；Fitts与选择BPS的公式不同，不可直接互换。

## 图 4：Online Grid Behavior

### 计算公式

**符号**

- W为累计60秒有效尝试时间窗口，T=60秒，C为窗口内完成的正确选择数，N为已核实的目标数。排除ITI、block休息和反馈等待，保留选择dwell、失败及超时耗时。

**步骤**

1. 按累计有效时间每1秒更新；长run从第60秒开始，整个run不足60秒时用一次实际全程窗口，ratio=T/60，BPS=bits/(60×ratio)。窗口左边界切入trial时只累计重叠时间，选择结局按完成时刻落在窗口内计数。不跨session、录制片段、缺失计时或trial编号断档建窗。
2. Grid仅画C×log₂(N)/60。CO8保留Neuralink式(C−明确错误选择)×log₂(7)/60和Wolpaw估算；Fitts使用同一60秒窗口内完整成功trial的难度总和/外向运动总时长，边界不完整trial不参与Fitts。
3. 每run的median/IQR和peak来自60秒、step 1秒窗口；仅整个run不足60秒时按实际时长折算，峰值表记录有效时间与trial范围。IQR表示窗口分布，不是置信区间。

### 结果解读

- **行为与BPS**：成功率与到靶时间继续按各图标明的block或run汇总。BPS曲线横轴为累计有效尝试分钟；run汇总的每个点对应一个run。Grid采用横向A4比例的上下两个子图，共用run横轴。上图为蓝色BPS median/IQR与橙色peak，下图为橘色成功trial到靶时间median/IQR（秒）。到靶时间按同run原始成功trial汇总，不平均滚动窗口的RT。绿色虚线为理想100%成功率下达到6 bits/s所需的平均TTT=log₂(N)/6，不是成功trial中位数的严格阈值；存在失败、超时耗时时需更快。

### 质量评价标准

- **行为与BPS**：整个run不足60秒时按实际时长折算；网格大小或Fitts输入无法核实时显示N/A，不把不完整开头当作峰值。超时不能删除。滚动窗口重叠，不是独立样本；Fitts与选择BPS的公式不同，不可直接互换。

## 图 5：Online Grid Run Rolling BPS Summary

### 计算公式

**符号**

- W为累计60秒有效尝试时间窗口，T=60秒，C为窗口内完成的正确选择数，N为已核实的目标数。排除ITI、block休息和反馈等待，保留选择dwell、失败及超时耗时。

**步骤**

1. 按累计有效时间每1秒更新；长run从第60秒开始，整个run不足60秒时用一次实际全程窗口，ratio=T/60，BPS=bits/(60×ratio)。窗口左边界切入trial时只累计重叠时间，选择结局按完成时刻落在窗口内计数。不跨session、录制片段、缺失计时或trial编号断档建窗。
2. Grid仅画C×log₂(N)/60。CO8保留Neuralink式(C−明确错误选择)×log₂(7)/60和Wolpaw估算；Fitts使用同一60秒窗口内完整成功trial的难度总和/外向运动总时长，边界不完整trial不参与Fitts。
3. 每run的median/IQR和peak来自60秒、step 1秒窗口；仅整个run不足60秒时按实际时长折算，峰值表记录有效时间与trial范围。IQR表示窗口分布，不是置信区间。

### 结果解读

- **行为与BPS**：成功率与到靶时间继续按各图标明的block或run汇总。BPS曲线横轴为累计有效尝试分钟；run汇总的每个点对应一个run。Grid采用横向A4比例的上下两个子图，共用run横轴。上图为蓝色BPS median/IQR与橙色peak，下图为橘色成功trial到靶时间median/IQR（秒）。到靶时间按同run原始成功trial汇总，不平均滚动窗口的RT。绿色虚线为理想100%成功率下达到6 bits/s所需的平均TTT=log₂(N)/6，不是成功trial中位数的严格阈值；存在失败、超时耗时时需更快。

### 质量评价标准

- **行为与BPS**：整个run不足60秒时按实际时长折算；网格大小或Fitts输入无法核实时显示N/A，不把不完整开头当作峰值。超时不能删除。滚动窗口重叠，不是独立样本；Fitts与选择BPS的公式不同，不可直接互换。

## 图 6：Online Grid Run Time to Target

### 计算公式

**符号**

- RT = 成功命中时刻 − trial开始时刻，包含选择dwell，单位秒。

**步骤**

1. 按run汇总原始成功trial的RT，不重复计入滚动窗口中的trial。计算median、Q1、Q3，橘色单轴绘制并标出median。

### 结果解读

- **到靶时间**：圆点及数值为run的成功trial中位到靶时间，误差棒为Q1–Q3。

### 质量评价标准

- **到靶时间**：失败和超时不作为成功trial的RT；没有可核实命中时间时显示N/A。IQR不是置信区间。
