# Link 2 Studio Control

本项目是一个本地浏览器控制台，提供 UVC 视频回流、本地录制和云台摇杆。

## 运行

在 Windows 的项目目录执行 `node server.js`，然后打开 `http://localhost:5173`。连接 Insta360 Link 2 后可直接开始录制。本机服务通过 Windows 摄像头驱动发送云台指令。摇杆的水平和垂直轴均采用反向拖拽，松手发送停止指令。

## 本机 API 与 Agent 技能

在 `bci_tools` 启动台刷新后选择 **Insta360 Link 2 Control**，启动后会打开浏览器。也可在本目录运行 `node server.js --open`。Node.js 必须可从 PATH 找到。无设备时可运行 `powershell -File examples/check-api.ps1` 验证接口。连接设备后，实机检查包括打开预览、在画面上滚动观察倍率、短推摇杆后松手，以及录制并确认浏览器下载生成的视频。

来源为 [Bertie97/Insta360-link2-control](https://github.com/Bertie97/Insta360-link2-control) 的 1.0.0 源码快照，提交 `e016c04`。Windows GitHub Actions 在工具目录或其工作流变更时运行，USB 设备检查在本机进行。

需要 Windows、Node.js 20 或更新版本、Windows PowerShell 5.1 和 USB 连接的 Link 2。运行 `npm start` 启动服务，再运行 `node link2.js lens` 查询镜头。执行 `npm run check` 可验证无需摄像头的路由和输入检查。

参见 [Link 2 技能](skills/insta360-link2-control/SKILL.md) 和 [HTTP API](skills/insta360-link2-control/references/api.md)。预览及录制通过浏览器操作。

## 云台协议

连接型号为 `2E1A:4C04` 的 Insta360 Link 2 后，Windows 本机服务通过 `IKsControl` 枚举摄像头的 UVC 扩展节点并访问属性 `0x16`。USB 捕获中，与官方程序云台日志逐条对应的是 `SET_CUR`、`wValue=0x1600`、`wIndex=0x0900`，四个数据字节依次为水平​​方向、水平速度、俯仰方向、俯仰速度。两个方向字段都为零时停止移动。Windows 标准相对云台属性的探测未通过，已实测可用的路径是 UVC 扩展属性。

## 实机验证（2026-11-27）

验证程序向页面摇杆使用的本机端点发送速度为 5、约 130 毫秒的脉冲，每次随后发送停止指令。验证直接读取实体 Link 2 的视频，对比缩放至 640 × 360 的前后画面并估计固定背景位移。

| 指令 | 水平位移（像素） | 垂直位移（像素） |
| --- | ---: | ---: |
| 静止 | 0.0 | 0.0 |
| 水平正向 | -16.4 | -0.2 |
| 水平反向 | 18.5 | 0.8 |
| 俯仰正向 | 0.1 | -10.4 |
| 俯仰反向 | 0.0 | 17.6 |

上述测试覆盖本机端点到摄像头画面的链路。录制文件保存在电脑上。
