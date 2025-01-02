##### 将python打包成可执行exe

将python打包的方式大概分为2种，其中每一种都可以简单打包和压缩打包（压缩打包就是构件单独的环境，将项目不需要的包隔离出去）

1. 单个文件的打包
2. 多个文件的打包（当面对一个大项目的时候，为了方便维护，通常将代码分到不同的文件中。多个文件同时打包的时候，就可以避免原代码泄露）

**1.单个文件的打包**

**简单打包：**不创建虚拟环境进行打包，这样会把一些不相关的包打包进去，导致打包文件较大，不推荐。

打开安装好的Anaconda Prompt（用管理员命令打开）。
使用cd命令移动到存放py文件的文件夹下。

基础打包（生成一个文件夹）：

```
pyinstaller converter.py
```

打包成单个文件：

```
pyinstaller --onefile converter.py
```

打包成单个文件且不显示控制台窗口（最推荐）：

```
pyinstaller --onefile --windowed converter.py
```

如果你想添加图标：

```
pyinstaller --onefile --windowed --icon=your_icon.ico converter.py
```

如果你想让生成的文件更小，可以使用以下命令：

```
pyinstaller --onefile --windowed --clean --noupx converter.py
```

**单独虚拟环境打包**：创建一个单独的虚拟环境，只将文件需要的包进行打包。打包之后的exe相对较小

打包过程和上述一样，在打包之前，多了一个创建单独虚拟环境的步骤

创建虚拟环境：

```
conda create -n “name”
```

安装相关的包

```
pip install ”name“
```

再进行上述的打包就可以了。

**2.打包多个文件**

待更新