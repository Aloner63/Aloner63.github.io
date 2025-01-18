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
