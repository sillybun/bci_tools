# 验证记录

验证日期：2026-10-28。环境：Linux x64、Node.js 24.19.0。

- `npm test`：4 项模型测试通过。
- `npm run test:ui`：5 项 Chromium 交互测试通过，覆盖新建、编辑、持久化、导入导出、左右边框、Ctrl / Command 自由调整、自动分行、类别边沿、跨类别移动、取消拖动、缩放和平移。
- `npm run build`：生产构建通过。
- `npm audit`：0 个已知漏洞。
- `npm run dist -- --linux --publish never`：AppImage / deb 均成功生成。
- 打包后的 Electron 45.1.0 Linux 程序：用独立临时数据目录启动，成功加载 5 个示例任务，打开编辑器并保存任务名称。自动化启动使用 `--no-sandbox` 适配测试环境；产品配置未禁用 sandbox。
- 手动查看浏览器截图，已修复隐藏导入控件显示、单行类别说明溢出问题。

Windows / macOS 的原生安装包构建尚未在本机执行；由 `.github/workflows/build.yml` 在对应原生 runner 上生成。安装包尚未配置商业签名及公证。

## 1.1.0 增量验证

- `npm test`：10 项通过。新增依赖图循环保护、前后联动、多前置与菱形依赖、空行删除、旧版项目兼容、SVG 输出校验。
- `npm run test:ui`：9 项通过。新增 Shift 连接与取消、关键点创建/编辑/删除/拖动及 Ctrl 自由调整、图中直接删除空行、下拉菜单 JSON/PNG/SVG 导出。
- SVG 经 XML 解析验证；PNG 经文件头及像素尺寸验证，并实际查看导出图片排版。
- `npm run dist -- --linux --publish never`：1.1.0 AppImage / deb 成功生成。
- 打包后的桌面程序：关键点拖动正常；Electron 原生 `will-download` 事件验证 PNG 保存完成（261803 字节），浏览器测试之外的桌面下载流程已确认。
- `npm install --package-lock-only` 的依赖审计：0 个已知漏洞。
- UI 截图见 `docs/screenshot.png`、`docs/export-menu.png`；图片导出样例见 `docs/export-example.png`。

## 1.2.0 增量验证

- `npm test`：12 项通过，覆盖项目库旧版迁移、保存失败保护、名称与显示设置持久化。
- `npm run test:ui`：11 项通过，覆盖双击改名、新建/切换项目、重启恢复、独立显示设置，以及 PNG/JPG/SVG 与设置同步。
- JPG 通过 JPEG 文件头、浏览器图片解码和桌面 Chromium 下载流程验证。
- `npm run dist -- --linux --publish never`：1.2.0 AppImage / deb 成功生成。

## 1.3.0 增量验证

- 14 项模型测试、13 项浏览器测试通过，覆盖撤销/重做、分支失效、历史上限、项目隔离、拖动整体恢复、输入框原生行为及连线颜色。
- 实际查看 PNG：图例使用绘制的虚线/空心箭头，关键点文字加大，连线继承起始类别颜色。
- Linux AppImage / deb 构建成功，本次使用已缓存的 Electron 45.1.0 zip，避免下载网络问题。
- 打包后桌面程序的原生菜单撤销/重做经自动化执行并验证状态恢复。
