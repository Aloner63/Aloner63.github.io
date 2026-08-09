##### 将python打包成可执行exe

将python打包的方式大概分为2种，其中每一种都可以简单打包和压缩打包（压缩打包就是构件单独的环境，将项目不需要的包隔离出去）

1. 单个文件的打包
2. 多个文件的打包（当面对一个大项目的时候，为了方便维护，通常将代码分到不同的文件中。多个文件同时打包的时候，就可以避免原代码泄露）

**1.单个文件的打包**

**单独虚拟环境打包**：创建一个单独的虚拟环境，只将文件需要的包进行打包。打包之后的exe相对较小

创建虚拟环境：

```
conda create -n “name” python=3.9
```

安装相关的包(程序中用到的包)

```
pip install ”name“
```

再进行打包就可以了。

打包成单个文件且不显示控制台窗口：

```
pyinstaller --onefile --windowed converter.py
```

如果你想添加图标：

```
pyinstaller --onefile --windowed --icon=your_icon.ico converter.py
```

如果你想让生成的文件更小，可以使用以下命令：
（这条命令的作用是将 converter.py 脚本打包成一个单独的、无控制台窗口的可执行文件，并在打包过程中不使用 UPX 压缩，同时清理掉旧的构建文件。这样生成的可执行文件适合用于 GUI 应用程序，并且在某些情况下可能更稳定（因为没有使用 UPX 压缩））
```
pyinstaller --onefile --windowed --clean --noupx converter.py
```
**2.打包多个文件**

默认情况下，PyInstaller 会创建一个包含多个文件的输出，而不是单个 .exe 文件。您只需不使用 --onefile 选项即可。

会在 dist 目录中创建一个名为 TSS的文件夹，其中包含所有必要的文件和库。代码中使用到的库不会被打包到exe中，这大大减少了exe的体积。
**注意：保证你打包的.py为程序的入口**
（无自定义图标）
```
pyinstaller --name=TSS --windowed main.py
```

（有自定义图标）

```
pyinstaller --name=TSS --windowed --clean --icon=your_icon.ico main.py
```


---

上述打包方式，只会在exe上显示图标。并不会在程序运行时，在状态栏和程序头部显示图标。以下是实现上述缺陷的办法。

1，首先在 ThymomaSegmentationApp 类中添加图标设置:

（程序运行时，显示在头部）

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/image-20250118204903403.png)

```python
# thymoma_segmentation_app.py
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QLabel, QProgressBar

class ThymomaSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.output_dir = None
        self.image_files = []
        self.current_index = 0
        self.zoom_factor = 1.0
        
        # 设置应用程序图标
        icon_path = "path/to/your/icon.ico"  # 替换为您的图标路径
        self.setWindowIcon(QIcon(icon_path))  # 设置窗口图标
        
        self.initUI()
        self.loadModel()

    def initUI(self):
        # 现有的 initUI 代码...
```

2，在打包时设置图标:

```
pyinstaller --name=ThymomaSegmentation ^
            --windowed ^
            --icon=path/to/your/icon.ico ^
            --add-data "path/to/your/icon.ico;." ^
            main.py
```

3，在主程序入口点设置任务栏图标:

（保证任务栏显示图标）

```python
# main.py
import sys
import ctypes
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from thymoma_segmentation_app import ThymomaSegmentationApp

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序 ID（Windows）
    if sys.platform == 'win32':
        myappid = 'company.product.subproduct.version'  # 自定义的应用程序 ID
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    # 设置应用程序图标
    app_icon = QIcon("path/to/your/icon.ico")
    app.setWindowIcon(app_icon)
    
    window = ThymomaSegmentationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```

4，为了确保图标文件被正确打包和加载，您可以创建一个资源文件:

```python
# resources.py
import os
import sys

def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 创建临时文件夹，将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)
```

5，然后在应用程序中使用这个函数:

```python
# thymoma_segmentation_app.py
from resources import get_resource_path

class ThymomaSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # ...
        
        # 使用资源路径加载图标
        icon_path = get_resource_path("icon.ico")
        self.setWindowIcon(QIcon(icon_path))
        
        # ...
```

6，如果您想要更完整的图标支持，可以创建一个 .rc 文件:

```
// app.rc
IDI_ICON1               ICON    DISCARDABLE     "icon.ico"
```

然后使用 pyrcc5 将其编译为 Python 模块：

```
pyrcc5 app.rc -o app_rc.py
```

7，最后，在打包时包含所有必要的文件:

```python
pyinstaller --name=ThymomaSegmentation ^
            --windowed ^
            --icon=icon.ico ^
            --add-data "icon.ico;." ^
            --add-data "app_rc.py;." ^
            main.py
```

完整目录结构

your_project/
│
├── main.py
├── thymoma_segmentation_app.py
├── resources.py
├── icon.ico
├── app.rc
├── app_rc.py
└── utils/
    └── utils.py

perfect！
