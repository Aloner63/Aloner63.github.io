#### vscode运行c++

###### 在c++的编译中，g++，gdb，gcc都是什么？

**GCC (GNU Compiler Collection)**，这是GNU项目的编译器集合，最初是"GNU C Compiler"的缩写，现在支持多种编程语言，**主要用于编译C语言程序**，命令格式：gcc source.c -o output

**G++ (GNU C++ Compiler)**是GCC的一部分，**专门用于编译C++程序**，会自动链接C++标准库，将.cpp文件视为C++源代码（而gcc默认将其视为C文件），命令格式：g++ source.cpp -o output

**GDB (GNU Debugger)**是GNU项目的调试器，用于程序调试，可以：（设置断点，单步执行，查看变量值，查看调用栈，监控变量变化）命令格式：gdb ./program

GNU项目（GNU Project）是一个非常重要的自由软件运动。其与Linux的关系是：Linux内核 + GNU工具 = 完整的操作系统。严格来说，我们常说的"Linux"实际上是"GNU/Linux"系统。



准备工作，下载好vscode安装包和MinGW-W64安装包

1. vscode的安装
2. 安装MinGW-W64及配置环境变量

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/145416d97b141096571d8073ada18314.png)


##### 单个文件的编译和运行

单个文件直接根据vscode编译运行即可。（会自动生成一个task.json文件）

写好测试代码后，点击右上角的调试按钮，这时会弹出调试程序的选项，选择第一个，也就是 gcc 编译工具。

```c
#include <stdio.h>

int main()
{
    for (int i = 0; i < 5; i++)
        printf("Hello Grayson~%d\n", i); 

    return 0;
}

```

##### 多个.c文件的运行和调试

当需要多个.c文件需要编译和运行的时候。就需要task.json文件。

这个文件是用于定义任务配置，这些任务可以在 VS Code 中运行，例如编译代码、运行测试、启动调试器等。tasks.json文件是一个 JSON 格式的文件，其中包含了任务的配置信息，包括任务名称、命令、参数等。通过编辑tasks.json文件，我们可以自定义项目中的各种任务，并在 VS Code 中方便地执行这些任务。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/70c6c0e46257925345a57c76e5230b6a.png)

具体修改如下图所示，我注释掉了原来的"${file}"，并新增一行"*.c"，表示并非指定某一个 .c 文件，而是当前文件夹下所有的 .c 文件。同时也把"${fileDirname}\\${fileBasenameNoExtension}.exe"注释掉，改成"${fileDirname}\\program.exe"，那么多个 .c 文件编译之后的可执行文件就是program.exe。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/d87b4f522dc14d9a6cbd15370fbd16a0.png)

修改好后，按组合键`Ctrl + s`保存即可。

还需要一个launch.json文件。点击左侧的运行和调试，再点击创建lanuch.json文件。

搜索框会弹出选项，选择c++（GDB/LLDB）

然后 VS Code 会新建一个 JSON 文件，点击右下角的`添加配置`，在弹出的下拉菜单中选择`C/C++：（gdb）启动`。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/d80525fca1e08143dc63379be33dd9a9.png)

此时，JSON 文件会多出一些配置信息，需要我们修改的内容如下图所示的红框标志内容。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/2a4329d95fdf853dc91c85a605c10f28.png)

修改为下图红框所示内容，`“program”`后的内容就是前面提到的`tasks.json`文件中的编译后产生的可执行文件。`"miDebuggerPath"`后面的则是前面安装的 MinGW-W64 的 gdb 工具的路径。修改后保持关闭。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/47664b98a346762eb2f10e45513cecba.png)



代码测试

之后，我们进行多文件的编译调试，先在`C`文件夹下新建一个新的文件夹，我这里命名为`test2`，并在这个文件夹里面新建三个文件，分别是`test.c`、`max.h`和`max.c`。

![](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/a711137d2a24a25241d9ea639e337434.png)

代码如下：

max.h

```c
#ifndef __MAX_H__
#define __MAX_H__
#include <stdio.h>

int findMaxNum(int num1, int num2);

#endif // __MAX_H__

```

max.c

```c
#include "max.h"

int findMaxNum(int num1, int num2)
{
    return num1 > num2 ? num1 : num2;
}

```

test2.c

```c
#include <stdio.h>
#include "max.h"

int main()
{
    int a = 10;
    int b = 20;
    int c = findMaxNum(a, b);
    printf("%d\n", c);
    return 0;
}
```

代码写好后，给`test2.c`的第 8 行代码打一个断点，再点调试按钮旁边的小三角形，在下拉菜单中选择`调试C/C++文件`。

如果点击`继续`，调试过程会跳到下一个断点，不过我们这个程序只打了一个断点，所以会直接运行到程序结束并退出调试。

如果点击`逐过程`，则在不进入函数内部，而是直接输出函数的运行结果，然后跳到下一行。

如果点击`单步调试`，则会进入被调用函数的内部，继续点击`单步调试`会一步一步执行并返回。如果进入函数后，点击`单步跳出`则直接带着函数的执行结果返回被调用处。

------

这时就可以编译和运行多个c文件和多个c++文件了。

