**从根本上来说，vscode就是一个文本编译器。**它之所以能狗对多种代码进行编写，本质上还是通过插件，各种包的下载和相关环境的配置实现的。

简单方法（小工程）
在软件界面按住Ctrl+Shift+P，在软件上方出现输入框中输入：C/C++:Edit Configurations。选择第一个配置JSON配置。
完成之后，.vscode目录下面会自动添加了一个c_cpp_properties.json。在c_cpp_properties.json中加入自己需要的头文件路径。
注意：不光只添加自己所需的头文件路径，还要添加include等等路径
```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/lock",
                "/usr/include",
                "/usr/include/c++/11",
                "/usr/include/x86_64-linux-gnu/c++/11"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "gnu++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```


**第一步**
卸载老的vscode（如果电脑上没有安装过，跳过）
电脑上如果已经有了vscode的，想重新安装的。先将vscode完全卸载。去源文件夹中找到unins000.exe卸载。
也可用geek（卸载神器）卸载。卸载之后可能还是会卸载不干净，这时候我们到c盘用户文件夹下，将.vscode文件夹删除。接下来，我们打开AppData文件夹，点击其中的Roaming文件夹并打开，删除其中的code和vscode文件夹，从而完成彻底的卸载，之后再进行安装就相当于是安装全新的VSCode了

**第二步**
安装新的vscode
[官网下载](https://code.visualstudio.com/docs/cpp/config-mingw)，无脑下一步。安装完成。
打开vscode。作为一个英语残废，当然是先安装一个中文语言包了。之后安装c/c++扩展，这时候不需要将vscode关闭，最小化即可。

**第三步**
安装编译器
使用MSYS2下载安装MinGW-x64编译器。[MSYS2](https://www.msys2.org/)
安装完成后，到电脑的开始菜单中找到**MSYS2 MSYS**。输入**pacman -Su**指令以更新余下的库，当过程中询问是否继续安装时，输入Y即可。
目前为止，MSYS2已经配置完毕。
下面下载ming-w64 GCC来进行编译，输入指令**pacman -S --needed base-devel mingw-w64-x86_64-toolchain**进行下载，当过程中询问是否继续安装时，输入Y即可。
下载完成后，打开msys2的文件夹，看是否有mingw文件夹，如果有，就证明成功下载了GCC编译器。
打开命令行，用**g++ --version, gdb --version**来查看相应的版本（这个地方可能需要重启后才能起作用）

将编译器添加到环境变量的PATH中。以确保VSCode有找到编译器的路径。找到之前下载后msys64文件夹内mingw64中bin文件的路径，复制并输入，一路点击确定，关闭所有窗口。
在命令行中使用**gcc -v**或**g++ -v**验证，如果出现相应的反馈，则证明路径添加正确。


**第四步**
vscode的配置
打开vscode随便创建一个cpp文件，直接build。**单击Terminal(终端)->Configure Default Build Task**
随后在跳出的选择方框中点击C/C++: g++.exe build active file 以编译 helloworld.cpp 并创建可执行文件helloworld.exe
这样，我们会发现在左侧的文件夹栏目中出现了.vscode文件夹，其中含有tasks.json文件，我们打开该文件，并输入如下代码：
```
{
  "tasks": [
    {
      "type": "cppbuild",
      "label": "C/C++: g++.exe build active file",
      "command": "D:/msys64/mingw64/bin/g++.exe",
      "args": ["-g", "${file}", "-o", "${fileDirname}\\${fileBasenameNoExtension}.exe"],
      "options": {
        "cwd": "${fileDirname}"
      },
      "problemMatcher": ["$gcc"],
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "detail": "compiler: D:/msys64/mingw64/bin/g++.exe"
    }
  ],
  "version": "2.0.0"
}
```
注意：根据自己的路径进行相应的修改

接下来，我们进一步进行配置。该配置是针对于程序运行的，点击Run（运行）->Add Configuration（添加配置），这样会在.vscode文件夹中生成launch.json文件
```
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "g++.exe - Build and debug active file",
      "type": "cppdbg",
      "request": "launch",
      "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${fileDirname}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "miDebuggerPath": "D:\\msys64\\mingw64\\bin\\gdb.exe",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ],
      "preLaunchTask": "C/C++: g++.exe build active file"
    }
  ]
}
```
一样的，根据自己的实际情况进行相应的路径修改。

接下来，我们进行最后一项的配置。该配置时针对于C/C++的相关拓展的，我们首先按下Ctrl+Shift+P以打开全局搜索，输入C/C++并在搜索结果中选择C/C++: Edit configurations(UI)
点击该选项，我们会打开一个设置的图形界面，将其中Configuration name(配置名称)下的Select a configuration set to edit(选择要编辑的配置集)下的文本框，输入GCC；在IntelliSense mode(IntelliSense 模式)下的文本框中选择windows-gcc-x64。
在进行上述操作之后，我们会发现在.vscode文件夹中出现了c_cpp_properties.json文件，打开后：
```
{
  "configurations": [
    {
      "name": "GCC",
      "includePath": ["${workspaceFolder}/**"],
      "defines": ["_DEBUG", "UNICODE", "_UNICODE"],
      "windowsSdkVersion": "10.0.18362.0",
      "compilerPath": "D:/msys64/mingw64/bin/g++.exe",
      "cStandard": "c17",
      "cppStandard": "c++17",
      "intelliSenseMode": "windows-gcc-x64"
    }
  ],
  "version": 4
}
```
根据自己的实际情况进行修改。


**扩展推荐：**
C/C++ Extension Pack：一些常用的C/C++拓展
Code Runner：代码运行器
Tabnine：AI自动代码填充（外网代理登陆后，切换内网即可使用）