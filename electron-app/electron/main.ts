import { app, BrowserWindow } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const isDev = !app.isPackaged;

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      contextIsolation: true,
    },
  });

  if (isDev) {
    console.log('[DEV] Loading localhost:5173');
    win.loadURL('http://localhost:5173');
  } else {
    console.log('[PROD]❤️ Loading local index.html');
    win.loadFile(path.join(app.getAppPath(), './dist/index.html'));
  }
  
}

app.whenReady().then(createWindow);
