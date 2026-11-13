const { app, BrowserWindow, Menu } = require("electron");
const path = require("node:path");
function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 850,
    minHeight: 600,
    backgroundColor: "#f5f7fb",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e) => e.preventDefault());
  win.loadFile(path.join(__dirname, "../dist/index.html"));
}
app.whenReady().then(() => {
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: "编辑",
        submenu: [
          {
            label: "撤销",
            accelerator: "CmdOrCtrl+Z",
            click: () =>
              BrowserWindow.getFocusedWindow()?.webContents.executeJavaScript(
                "window.dispatchEvent(new CustomEvent('gantt-history',{detail:'undo'}))",
              ),
          },
          {
            label: "重做",
            accelerator: "CmdOrCtrl+Shift+Z",
            click: () =>
              BrowserWindow.getFocusedWindow()?.webContents.executeJavaScript(
                "window.dispatchEvent(new CustomEvent('gantt-history',{detail:'redo'}))",
              ),
          },
          { type: "separator" },
          { role: "cut" },
          { role: "copy" },
          { role: "paste" },
          { role: "selectAll" },
        ],
      },
      {
        label: "窗口",
        submenu: [
          { role: "reload" },
          { role: "togglefullscreen" },
          { role: "close" },
        ],
      },
    ]),
  );
  createWindow();
  app.on("activate", () => {
    if (!BrowserWindow.getAllWindows().length) createWindow();
  });
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
